#coding: UTF-8

import sys
from copy import deepcopy
import csv

import numpy as np
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim

from hybridkg import HybridDistMult, Variable, ifcuda
from entity import Entity
from pca_data import PCA_data

with_kg = True

def load_dataset():
    with open("Countries/triples.txt") as h:
        reader = csv.reader(h, delimiter=" ")
        triples = [tuple([int(v) for v in row]) for row in reader]

    with open("Countries/entities_with_data.txt") as h:
        reader = csv.reader(h, delimiter=" ")
        ent_with_data = [int(row[0]) for row in reader]

    with open("Countries/entities.txt") as h:
        reader = csv.reader(h, delimiter=" ")
        n_entity = len(list(reader))

    with open("Countries/relations.txt") as h:
        reader = csv.reader(h, delimiter=" ")
        n_relation = len(list(reader))

    temperature = np.loadtxt("Countries/temperature.txt").reshape((len(ent_with_data), 115, 12))
    rainfall = np.loadtxt("Countries/rainfall.txt").reshape((len(ent_with_data), 115, 12))

    return triples, ent_with_data, temperature, rainfall, n_entity, n_relation

def calc_n_batch(n_data, s_batch):
    return n_data // s_batch + (0 if n_data % s_batch == 0 else 1)

def get_batch(n_data, s_batch, i_batch):
    n_batch = calc_n_batch(n_data, s_batch)
    i_batch = i_batch % n_batch
    start = i_batch * s_batch
    end = min((i_batch + 1) * s_batch, n_data)
    return start, end

def train(model, optimizer, triples, positive_lookup, rainfall_y, rainfall_idx, temperature_y, temperature_idx):
    model.train()

    n_entity = model.models[0].get_n_entity()
    n_rainfall = model.models[1].get_n_entity()
    n_temperature = model.models[2].get_n_entity()

    entity_idx = np.arange(n_entity, dtype=int)

    # shuffle data
    triples = shuffle(triples)
    entity_idx = shuffle(entity_idx)
    rainfall_y, rainfall_idx = shuffle(rainfall_y, rainfall_idx)
    temperature_y, temperature_idx = shuffle(temperature_y, temperature_idx)

    # iterate for batches
    size_triple_batch = 50
    size_entity_batch = 50

    #tripleとentityで数が違うのでバッチ数が異なる
    n_triple_batch = calc_n_batch(len(triples), size_triple_batch)
    n_entity_batch = calc_n_batch(entity_idx.shape[0], size_entity_batch)
    n_rainfall_batch = calc_n_batch(rainfall_y.shape[0], size_entity_batch)
    n_temperature_batch = calc_n_batch(temperature_y.shape[0], size_entity_batch)

    n_batch = max(n_triple_batch, n_entity_batch, n_rainfall_batch, n_temperature_batch)

    for i_batch in range(n_batch):
        optimizer.zero_grad()

        #entity
        #lossについて、以下ではentityの個数に応じて重みづける
        loss_entity = 0

        #KGのentityに対してlossを計算するのはKGを考慮するときのみ
        if with_kg:
            start, end = get_batch(entity_idx.shape[0], size_entity_batch, i_batch)
            loss_z = model.models[0].loss_z(entity_idx[start:end])
            loss_entity += torch.mean(loss_z) * n_entity

        #rainfall
        start, end = get_batch(rainfall_y.shape[0], size_entity_batch, i_batch)
        loss_z = model.models[1].loss_z(rainfall_idx[start:end], rainfall_y[start:end])
        #loss_entity += torch.mean(loss_z) * n_rainfall
        loss_entity  += torch.mean(loss_z) * model.models[1].get_n_entity()

        #temperature
        start, end = get_batch(temperature_y.shape[0], size_entity_batch, i_batch)
        loss_z = model.models[2].loss_z(temperature_idx[start:end], temperature_y[start:end])
        #loss_entity += torch.mean(loss_z) * n_temperature
        loss_entity  += torch.mean(loss_z) * model.models[2].get_n_entity()

        #triple
        #tripleのlossもtripleの個数で重み付ける
        #TODO: entityとtripleを異なる重みでlossに反映させてもいいかもしれない loss(entity) + C * loss(triple)みたいな感じに
        if with_kg:
            loss_triple = 0

            start, end = get_batch(len(triples), size_triple_batch, i_batch)

            #positive
            heads = np.array([h for h,r,t in triples[start:end]], dtype=int)
            relations = np.array([r for h,r,t in triples[start:end]], dtype=int)
            tails = np.array([t for h,r,t in triples[start:end]], dtype=int)

            y = np.ones(heads.shape[0])

            loss_positive = model.calc_loss_y(heads, relations, tails, y)

            #negative
            heads = []
            relations = []
            tails = []
            for h,r,t in triples[start:end]:
                if np.random.uniform() < 0.5:
                    while True:
                        # replace head with entity from same class (or model)
                        h = (h[0], np.random.randint(model.models[h[0]].get_n_entity()))
                        if not ((h,r,t) in positive_lookup):
                            break
                else:
                    while True:
                        # replace tail with entity from same class (or model)
                        t = (t[0], np.random.randint(model.models[t[0]].get_n_entity()))
                        if not ((h,r,t) in positive_lookup):
                            break
                heads.append(h)
                relations.append(r)
                tails.append(t)

            heads = np.array(heads, dtype=int)
            relations = np.array(relations, dtype=int)
            tails = np.array(tails, dtype=int)

            y = np.zeros(heads.shape[0])

            loss_negative = model.calc_loss_y(heads, relations, tails, y)

            loss_triple = (torch.mean(loss_positive) + torch.mean(loss_negative)) * len(triples)

        #train one step
        if with_kg:
            loss = loss_entity + loss_triple
        else:
            loss = loss_entity
        loss.backward()
        optimizer.step()

        sys.stdout.write("training batch %d/%d\r"%(i_batch, n_batch))
        sys.stdout.flush()

def valid(model, model_idx, y, idx):
    model.eval()

    # temporary ignore KG terms
    bu = model.models[model_idx].with_kg
    model.models[model_idx].with_kg = False

    # calculate mean loss for test set
    loss = torch.mean(model.models[model_idx].loss_z(idx, y))
    loss = loss.data.cpu().numpy()[0]

    # reset PCA model configure
    model.models[model_idx].with_kg = bu

    return loss

def valid_kg(model, test_triples, remove_lookup):
    model.eval()

    ranks = []

    for i_triple, test_triple in enumerate(test_triples):
        sys.stdout.write("%d-%d\r"%(i_triple,len(test_triples)))
        sys.stdout.flush()

        h,r,t = test_triple

        negative_heads = []
        negative_tails = []
        for e in range(model.models[h[0]].get_n_entity()):
            if not (((h[0],e), r, t) in remove_lookup):
                negative_heads.append((h[0], e))
        for e in range(model.models[t[0]].get_n_entity()):
            if not ((h, r, (t[0], e)) in remove_lookup):
                negative_tails.append((t[0], e))

        true_score = model.calc_score(
            np.array([h], dtype=int),
            np.array([r], dtype=int),
            np.array([t], dtype=int)
        ).data.cpu().numpy()[0]

        neg_score1 = model.calc_score(
            np.array(negative_heads, dtype=int),
            np.array([r]*len(negative_heads), dtype=int),
            np.array([t]*len(negative_heads), dtype=int)
        ).data.cpu().numpy().tolist()

        neg_score2 = model.calc_score(
            np.array([h]*len(negative_tails), dtype=int),
            np.array([r]*len(negative_tails), dtype=int),
            np.array(negative_tails, dtype=int)
        ).data.cpu().numpy().tolist()

        scores = neg_score1 + [true_score]
        scores = np.array(scores)
        rank = np.sum(scores > true_score) + 1
        ranks.append(rank)

        scores = neg_score2 + [true_score]
        scores = np.array(scores)
        rank = np.sum(scores > true_score) + 1
        ranks.append(rank)

    ranks = np.array(ranks)

    MRR = np.sum(1.0/ranks)/ranks.shape[0]
    HIT3 = np.mean(ranks<4)

    return MRR, HIT3

def main(kg, seed, dim_emb, dim_latent, dim_hidden):
    global with_kg
    with_kg = kg

    triples, ent_with_data, temperature, rainfall, n_raw_entity, n_raw_relation = load_dataset()
    temperature /= 40. #normalize
    rainfall /= 3000. #normalize
    e2d = {e:i for i,e in enumerate(ent_with_data)} #entity id -> data idx
    #data idxは各国の降水量データにおける行番号

    # In triple, each entity is described as (model_idx, entity_idx_in_model)
    # model 0: normal entity.
    # model 1: PCA for rainfall data. (latent variable of country -> 12 dim monthly rainfall)
    # model 2: PCA for temperature data. (latent variable of country -> 12 dim monthly rainfall)
    # create new relation between entity and rainfall, temperature data.
    triples = [((0,h), r, (0,t)) for (h,r,t) in triples]
    for e in ent_with_data:
        d = e2d[e]
        triples.append(
            ((0,e), n_raw_relation, (1,d))
        )
        triples.append(
           ((0,e), n_raw_relation + 1, (2,d))
        )
    n_entity = n_raw_entity + len(ent_with_data)
    n_relation = n_raw_relation + 2

    #あるtripleが訓練データに含まれるかを検索するためのもの。set使うとそこそこ早い
    positive_lookup = set(triples)

    # create model
    # PCAは各國に1つ潜在変数が存在し、その潜在変数から実際のデータが1つ生成されるようなモデルとしている
    model = HybridDistMult(n_relation, dim_emb = dim_emb)
    ent_model = Entity(n_entity=n_raw_entity, dim_emb=dim_emb, if_reparam=False)
    ent_rainfall = PCA_data(n_entity=len(ent_with_data), dim_latent=dim_latent, dim_data=12,\
                                dim_hidden=dim_hidden, dim_emb=dim_emb)
    ent_temperature = PCA_data(n_entity=len(ent_with_data), dim_latent=dim_latent, dim_data=12,\
                                dim_hidden=dim_hidden, dim_emb=dim_emb)
    model.models.append(ent_model)
    model.models.append(ent_rainfall)
    model.models.append(ent_temperature)

    if ifcuda:
        model.cuda()

    if not with_kg:
        model.models[1].with_kg = False
        model.models[2].with_kg = False

    # set weight decay for relation parameter
    weight_decay_params = []
    non_weight_decay_params = []
    for name, param in model.named_parameters():
        print(name)
        if "relation" in name:
            weight_decay_params.append(param)
        else:
            non_weight_decay_params.append(param)
    optimizer = optim.Adam(
                                [
                                    {"params": weight_decay_params, "weight_decay":0.0001},
                                    {"params": non_weight_decay_params}
                                ],
                                lr=0.005, weight_decay=0
                             )

    #すべての国に対して115年分12次元のデータがあるので、何年ぶんかまとめてtrain, valid, testに分ける
    # rainfall data
    rainfall_idx = np.repeat(np.arange(len(ent_with_data), dtype=int)[:,np.newaxis], 115, axis=1)
    rainfall_idx, rainfall = shuffle(rainfall_idx, rainfall, random_state=seed)

    train_rainfall_y = rainfall[:,:40,:].reshape((-1,12))
    train_rainfall_idx = rainfall_idx[:,:40].flatten()

    valid_rainfall_y = rainfall[:,40:50,:].reshape((-1,12))
    valid_rainfall_idx = rainfall_idx[:,40:50].flatten()

    test_rainfall_y = rainfall[:,50:,:].reshape((-1,12))
    test_rainfall_idx = rainfall_idx[:,50:].flatten()

    #temperature data
    temperature_idx = np.repeat(np.arange(len(ent_with_data), dtype=int)[:,np.newaxis], 115, axis=1)
    temperature_idx, temperature = shuffle(temperature_idx, temperature, random_state=seed)

    train_temperature_y = temperature[:,:40,:].reshape((-1,12))
    train_temperature_idx = temperature_idx[:,:40].flatten()

    valid_temperature_y = temperature[:,40:50,:].reshape((-1,12))
    valid_temperature_idx = temperature_idx[:,40:50].flatten()

    test_temperature_y = temperature[:,50:,:].reshape((-1,12))
    test_temperature_idx = temperature_idx[:,50:].flatten()

    #
    best_valid_loss = 1e100
    best_test_loss = 1e100
    best_valid_loss_t = 1e100
    best_test_loss_t = 1e100
    best_epoch = 0
    for i_epoch in range(1000000):
        train(model, optimizer, triples, positive_lookup, train_rainfall_y, train_rainfall_idx, train_temperature_y, train_temperature_idx)
        print("")

        if i_epoch % 5 == 0:
            loss_valid = valid(model, 1, valid_rainfall_y, valid_rainfall_idx)
            loss_test = valid(model, 1, test_rainfall_y, test_rainfall_idx)
            #temperature
            loss_valid_t = valid(model, 2, valid_temperature_y, valid_temperature_idx)
            loss_test_t = valid(model, 2, test_temperature_y, test_temperature_idx)
            print(loss_valid_t, loss_test_t)
            if loss_valid < best_valid_loss:
                best_valid_loss = loss_valid
                best_test_loss = loss_test
                best_epoch = i_epoch
            print(i_epoch, loss_valid, loss_test)

            # #DEBUG:
            # if with_kg:
            #     print(valid_kg(model, test_triples, remove_lookup))

        if best_epoch < i_epoch - 50:
            break
    return best_valid_loss, best_test_loss, best_epoch

if __name__=="__main__":
    #mod3: 単純な線形写像。　temperatureモデルも導入
    #mod4: PCA側のlossについて変数の個数を国の個数とする

    h = open("exp_with_kg_T40_E100_mod4_2.log", "w")

    for dim_emb in [20,40,60,80]:
        for dim_latent in [3]:
            for dim_hidden in [10]:
                b_vs = []
                b_ts = []
                b_es = []
                for i in range(100):
                    b_v, b_t, b_e = main(True, None, dim_emb=dim_emb, dim_latent=dim_latent, dim_hidden=dim_hidden)
                    b_vs.append(b_v)
                    b_ts.append(b_t)
                    b_es.append(b_e)
                h.write("%s\n"%(" ".join([str(x) for x in b_es])))
                h.write("%d-%d-%d %.4f(%.4f) %.4f(%.4f)\n"%(dim_emb, dim_latent, dim_hidden, np.mean(b_vs), np.std(b_vs), np.mean(b_ts), np.std(b_ts)))
                h.flush()
    h.close()

    # h = open("exp_without_kg_T40_mod1.log", "w")
    #
    #
    # for dim_latent in [3]:
    #     b_vs = []
    #     b_ts = []
    #     b_es = []
    #     for i in range(10):
    #         b_v, b_t, b_e = main(False, None, dim_emb=1, dim_latent=dim_latent, dim_hidden=1)
    #         b_vs.append(b_v)
    #         b_ts.append(b_t)
    #         b_es.append(b_e)
    #     h.write("%s\n"%(" ".join([str(x) for x in b_es])))
    #     h.write("%d %.4f(%.4f) %.4f(%.4f)\n"%(dim_latent, np.mean(b_vs), np.std(b_vs), np.mean(b_ts), np.std(b_ts)))
    #     h.flush()
    # h.close()
