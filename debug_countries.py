#coding: UTF-8

import sys
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from hybridkg import Entity, HybridDistMult

def loadCountry():
    h = open("countries.json")
    tmpstr = h.read()
    h.close()

    countries = eval(tmpstr.replace("false", "False").replace("true", "True").replace("null", "None"))

    triples = []
    for country in countries:
        name = country["cca3"]
        region = country["region"]
        subregion = country["subregion"]
        borders = country["borders"]

        if region in ["Asia", "Europe"]:
            region = "Eurasia"

        triples.append((name, "isInside", region))
        triples.append((name, "isInside", subregion))
        triples.append((subregion, "isInside", region))
        for nei in borders:
            triples.append((name, "isNeighbor", nei))
    triples = list(set(triples))

    entities = list(set([h for h,r,t in triples] + [t for h,r,t in triples]))
    e2i = {e:idx for idx,e in enumerate(entities)}
    relations = list(set([r for h,r,t in triples]))
    r2i = {r:idx for idx,r in enumerate(relations)}

    train_triples = []
    test_triples = []
    while len(test_triples) < 100:
        triple = triples.pop(0)
        if triple[1] is "isNeighbor":
            test_triples.append(triple)
            test_triples.append((triple[2], triple[1], triple[0]))

            if (triple[2], triple[1], triple[0]) in triples:
                triples.remove((triple[2], triple[1], triple[0]))
        else:
            train_triples.append(triple)
    train_triples += triples

    train_triples = [(e2i[h], r2i[r], e2i[t]) for h,r,t in train_triples]
    test_triples = [(e2i[h], r2i[r], e2i[t]) for h,r,t in test_triples]

    return train_triples, test_triples, len(e2i), len(r2i)

def train(model, optimizer, entities, train_triples, train_lookup):
    model.train()

    size_triple_batch = 50
    n_triple_batch = len(train_triples) // size_triple_batch
    n_batch = n_triple_batch
    for i_batch in range(n_batch):
        i_triple_batch = i_batch % n_triple_batch

        optimizer.zero_grad()

        #triple
        start = i_triple_batch * size_triple_batch
        end = (i_triple_batch + 1) * size_triple_batch

        #positive
        heads = np.array([h for h,r,t in train_triples[start:end]], dtype=int)
        relations = np.array([r for h,r,t in train_triples[start:end]], dtype=int)
        tails = np.array([t for h,r,t in train_triples[start:end]], dtype=int)

        y = np.ones(heads.shape[0])

        loss_positive = model.calc_loss_y(heads, relations, tails, y)

        #negative
        heads = []
        relations = []
        tails = []
        for h,r,t in train_triples[start:end]:
            if np.random.uniform() < 0.5:
                while True:
                    h = entities[np.random.randint(len(entities))]
                    if not ((h,r,t) in train_lookup):
                        break
            else:
                while True:
                    t = entities[np.random.randint(len(entities))]
                    if not ((h,r,t) in train_lookup):
                        break
            heads.append(h)
            relations.append(r)
            tails.append(t)

        heads = np.array(heads, dtype=int)
        relations = np.array(relations, dtype=int)
        tails = np.array(tails, dtype=int)

        y = np.zeros(heads.shape[0])

        loss_negative = model.calc_loss_y(heads, relations, tails, y)

        #train one step
        loss = torch.mean(loss_positive) + torch.mean(loss_negative)
        loss.backward()
        optimizer.step()

        sys.stdout.write("batch %d/%d\r"%(i_batch, n_batch))
        sys.stdout.flush()

def valid(model, test_triples, entities, remove_lookup):
    model.eval()

    ranks = []
    for i_triple, test_triple in enumerate(test_triples):
        sys.stdout.write("%d-%d\r"%(i_triple,len(test_triples)))
        sys.stdout.flush()

        h,r,t = test_triple

        negative_heads = []
        negative_tails = []
        for e in entities:
            if not ((e,r,t) in remove_lookup):
                negative_heads.append(e)
            if not ((h,r,e) in remove_lookup):
                negative_tails.append(e)

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

if __name__=="__main__":
    train_triples, test_triples, n_entity, n_relation = loadCountry()

    thr_entity = n_entity // 2
    #thr_entity = n_entity - 1

    def sep(e):
        if e < thr_entity:
            return (0, e)
        else:
            return (1, e - thr_entity)
    train_triples = [(sep(h), r, sep(t)) for h,r,t in train_triples]
    test_triples = [(sep(h), r, sep(t)) for h,r,t in test_triples]

    entities = [(0,e) for e in range(thr_entity)] + [(1,e) for e in range(n_entity - thr_entity)]

    train_lookup = set(train_triples)
    remove_lookup = set(train_triples + test_triples)

    model = HybridDistMult(n_relation, dim_emb=20)
    ent1 = Entity(thr_entity, 20, if_reparam=False)
    ent2 = Entity(n_entity-thr_entity, 20, if_reparam=False)
    model.models.append(ent1)
    model.models.append(ent2)

    weight_decay_params = []
    non_weight_decay_params = []
    for name, param in model.named_parameters():
        print(name)
        if "relation" in name:
            weight_decay_params.append(param)
        else:
            non_weight_decay_params.append(param)
    optimizer = optim.Adagrad(
                                [
                                    {"params": weight_decay_params, "weight_decay":0.0001},
                                    {"params": non_weight_decay_params}
                                ],
                                lr=0.1, weight_decay=0
                             )

    for i_epoch in range(5000):
        train(model, optimizer, entities, train_triples, train_lookup)

        if i_epoch%5==0:
            MRR, HIT3 = valid(model, test_triples, entities, remove_lookup)

            print("")
            print("EPOCH::%d"%i_epoch)
            print(MRR, HIT3)
