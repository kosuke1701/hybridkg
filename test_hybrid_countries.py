#coding: UTF-8

import sys
from copy import deepcopy
import csv

import numpy as np
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim

from hybridkg import Entity, PCA_data, HybridDistMult, Variable

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

def train(model, optimizer, triples, positive_lookup, rainfall_y, rainfall_idx):
    model.train()

    n_entity = model.models[0].get_n_entity()
    n_rainfall = model.models[1].get_n_entity()

    entity_idx = np.arange(n_entity, dtype=int)

    # shuffle data
    triples = shuffle(triples)
    entity_idx = shuffle(entity_idx)
    rainfall_y, rainfall_idx = shuffle(rainfall_y, rainfall_idx)

    # iterate for batches
    size_triple_batch = 50
    size_entity_batch = 50

    n_triple_batch = calc_n_batch(len(triples), size_triple_batch)
    n_entity_batch = calc_n_batch(entity_idx.shape[0], size_entity_batch)
    n_rainfall_batch = calc_n_batch(rainfall_y.shape[0], size_entity_batch)

    n_batch = max(n_triple_batch, n_entity_batch, n_rainfall_batch)

    for i_batch in range(n_batch):
        optimizer.zero_grad()

        #entity
        loss_entity = 0

        if with_kg:
            start, end = get_batch(entity_idx.shape[0], size_entity_batch, i_batch)
            loss_z = model.models[0].loss_z(entity_idx[start:end])
            loss_entity += torch.mean(loss_z) * n_entity

        start, end = get_batch(rainfall_y.shape[0], size_entity_batch, i_batch)
        loss_z = model.models[1].loss_z(rainfall_idx[start:end], rainfall_y[start:end])
        loss_entity += torch.mean(loss_z) * n_rainfall

        #triple
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

def valid(model, rainfall_y, rainfall_idx):
    model.eval()

    # temporary ignore KG terms
    bu = model.models[1].with_kg
    model.models[1].with_kg = False

    # calculate mean loss for test set
    loss = torch.mean(model.models[1].loss_z(rainfall_idx, rainfall_y))
    loss = loss.data.cpu().numpy()

    # reset PCA model configure
    model.models[1].with_kg = bu

    return loss

if __name__=="__main__":
    triples, ent_with_data, temperature, rainfall, n_raw_entity, n_raw_relation = load_dataset()
    e2d = {e:i for i,e in enumerate(ent_with_data)} #entity id -> data idx

    # In triple, each entity is described as (model_idx, entity_idx_in_model)
    # model 0: normal entity.
    # model 1: PCA for rainfall data. (latent variable of country -> 12 dim monthly rainfall)
    # create new relation between entity and rainfall data with ID: n_raw_entity
    triples = [((0,h), r, (0,t)) for (h,r,t) in triples]
    for e in ent_with_data:
        d = e2d[e]
        triples.append(
            ((0,e), n_raw_relation, (1,d))
        )
    n_entity = n_raw_entity + len(ent_with_data)
    n_relation = n_raw_relation + 1

    positive_lookup = set(triples)

    # create model
    model = HybridDistMult(n_relation, dim_emb = 10)
    ent_model = Entity(n_entity=n_raw_entity, dim_emb=10, if_reparam=False)
    ent_rainfall = PCA_data(n_entity=len(ent_with_data), dim_latent=3, dim_data=12, dim_emb=10)
    model.models.append(ent_model)
    model.models.append(ent_rainfall)

    if not with_kg:
        model.models[1].with_kg = False

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

    # rainfall data
    rainfall_idx = np.repeat(np.arange(len(ent_with_data), dtype=int)[:,np.newaxis], 115, axis=1)

    train_rainfall_y = rainfall[:,:50,:].reshape((-1,12))
    train_rainfall_idx = rainfall_idx[:,:50].flatten()

    test_rainfall_y = rainfall[:,50:,:].reshape((-1,12))
    test_rainfall_idx = rainfall_idx[:,50:].flatten()


    for i_epoch in range(5000):
        train(model, optimizer, triples, positive_lookup, train_rainfall_y, train_rainfall_idx)
        print("")

        if i_epoch % 5 == 0:
            loss = valid(model, test_rainfall_y, test_rainfall_idx)
            print(i_epoch, loss)