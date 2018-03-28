#coding: UTF-8

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as _Variable

from log1exp import log1exp

ifcuda = False

def Variable(*args, **kwargs):
    var = _Variable(*args, **kwargs)
    if ifcuda:
        var = var.cuda()
    return var

def Cuda(tensor):
    if ifcuda:
        tensor = tensor.cuda()
    return tensor

class Entity(nn.Module):
    def __init__(self, n_entity, dim_emb, if_reparam):
        super(Entity, self).__init__()

        self.if_reparam = if_reparam
        self.dim_emb = dim_emb
        self.n_entity = n_entity

        self.mu = nn.Parameter(
            torch.FloatTensor(np.random.normal(size=(n_entity, dim_emb)))
        )
        if self.if_reparam:
            self.logvar = nn.Parameter(
                torch.FloatTensor(np.ones((n_entity, dim_emb)) * (-10.))
            )

    def encode(self, idx):
        if not isinstance(idx, _Variable):
            idx = Variable(torch.LongTensor(idx))

        if self.if_reparam:
            return self.mu[idx], self.logvar[idx]
        else:
            return self.mu[idx]

    def get_n_entity(self):
        return self.n_entity

    def loss_z(self, idx):
        u"""
        - q(z) { log p(z) - log q(z) }
        """
        if not isinstance(idx, _Variable):
            idx = Variable(torch.LongTensor(idx))

        if self.if_reparam:
            raise NotImplementedError("hoge")
        else:
            mu = self.mu[idx]
            l = 0.5 * 1.8378770 * self.dim_emb + 0.5 * torch.sum(mu ** 2, dim=1)
            return l

class PCA_data(nn.Module):
    def __init__(self, n_entity, dim_latent, dim_data, dim_emb, dim_hidden, with_kg=True):
        super(PCA_data, self).__init__()

        self.if_reparam = False
        self.n_entity = n_entity
        self.dim_latent = dim_latent
        self.dim_data = dim_data
        self.with_kg = with_kg

        self.x = nn.Parameter(
            torch.FloatTensor(np.random.normal(size=(n_entity, dim_latent)))
        )
        #NOTE: self.W is transposed version of W in equation y~WX+mu
        self.W = nn.Parameter(
            torch.FloatTensor(np.random.uniform(-1.,1.,size=(dim_latent, dim_data))/dim_latent**0.5)
        )
        self.mu = nn.Parameter(
            torch.FloatTensor(np.zeros(dim_data))
        )
        self.log_sigma2 = nn.Parameter(
            torch.FloatTensor(np.ones(1))
        )

        if with_kg:
            self.e = nn.Parameter(
                torch.FloatTensor(np.random.normal(size=(n_entity, dim_emb)))
            )

            #self.f_e_to_x = nn.Linear(dim_emb, dim_latent)
            self.f_e_to_x = nn.Sequential(
                nn.Linear(dim_emb, dim_hidden),
                nn.Tanh(),
                nn.Linear(dim_hidden, dim_latent)
            )
            self.log_V2 = nn.Parameter(
                torch.FloatTensor(np.ones((1)))
            )

    def encode(self, idx):
        if not self.with_kg:
            raise Exception("No KG mode.")
        if self.if_reparam:
            raise NotImplementedError("hoge")

        if not isinstance(idx, _Variable):
            idx = Variable(torch.LongTensor(idx))

        return self.e[idx]

    def get_n_entity(self):
        return self.n_entity

    def loss_z(self, idx, y):
        u"""
            y -- instance of Variable with size (n_entity, dim_data)
        """
        if self.if_reparam:
            raise NotImplementedError("hoge")

        if not isinstance(idx, _Variable):
            idx = Variable(torch.LongTensor(idx))
        if not isinstance(y, _Variable):
            y = Variable(torch.FloatTensor(y))

        #term log p(x)
        x = self.x[idx]
        l1 = 0.5 * 1.8378770 * self.dim_latent + 0.5 * torch.sum(x ** 2, dim=1)

        #term log p(y|x)
        C = self.W.t() @ self.W\
            + self.log_sigma2.exp() * Variable(torch.eye(self.dim_data))
        L = torch.potrf(C, upper=False)

        d = y - x @ self.W - self.mu
        L_inv_D_T, _ = torch.gesv(d.t(), L)

        #NOTE: L_ing_D_T's shape is (dim_data, n_idx), and we want to sum up for dim_data dimension.
        l2 = 0.5 * 1.8378770 * self.dim_data\
                + 0.5 * L.diag().log().sum()\
                + 0.5 * torch.sum(L_inv_D_T**2, dim=0)

        #term log p(e|x)
        if self.with_kg:
            mx = self.f_e_to_x(self.e[idx])

            l3 = 0.5 * 1.8378770 * self.dim_data\
                    + 0.5 * self.log_V2 * self.dim_data\
                    + 0.5 * torch.sum((mx - x)**2 / self.log_V2.exp(), dim=1)
        else:
            l3 = 0.

        return l1 + l2 + l3


class HybridDistMult(nn.Module):
    def __init__(self, n_relation, dim_emb):
        super(HybridDistMult, self).__init__()

        self.dim_emb = dim_emb
        self.if_reparam = True

        self.relation = nn.Parameter(
            torch.FloatTensor(n_relation, dim_emb).fill_(1.0)
        )
        self.models = nn.ModuleList([])

    def reparameterize(self, mu, logvar, mask):
        if self.if_reparam:
            std = logvar.mul(0.5).exp_() * mask
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def calc_z(self, nodes):
        u"""
        [Arguments]
            nodes -- ndarray (int) with size (*, 2).
                0th column contains model index for each node.
                1st column contains data index for each node.
        """
        n_model = len(self.models)
        n_data = nodes.shape[0]
        #data indices of node elements for each model
        data_idxs = [
            Variable(torch.LongTensor(nodes[np.equal(nodes[:,0], i_model)][:,1]))\
            if np.sum(np.equal(nodes[:,0], i_model))>0 else None for i_model in range(n_model)
        ]
        #indices of node elements for each model
        model_idxs = [
            Variable(torch.LongTensor(np.where(np.equal(nodes[:,0], i_model))[0]))\
            if np.sum(np.equal(nodes[:,0], i_model))>0 else None for i_model in range(n_model)
        ]
        #calculate latent variables (reparameterize if specified)
        mu = Variable(torch.FloatTensor(n_data, self.dim_emb).zero_())
        logvar = Variable(torch.FloatTensor(n_data, self.dim_emb).zero_())
        mask = Variable(torch.FloatTensor(n_data, 1).zero_())
        for i_model in range(n_model):
            data_idx = data_idxs[i_model]
            model_idx = model_idxs[i_model]
            if data_idx is None:
                continue
            model = self.models[i_model]
            ret = model.encode(data_idx)
            if model.if_reparam:
                _mu, _logvar = ret
                logvar[model_idx] *= _logvar
                mask[model_idx] += 1.
            else:
                _mu = ret
            mu[model_idx] += _mu
        z = self.reparameterize(mu, logvar, mask)

        return z

    def calc_score(self, heads, relations, tails):
        u"""
        [Arguments]
            heads, tails -- ndarray (int) with 2 dimensions.
                0th column contains model indices for each node.
                1st column contains data indices for each node.
            relations -- ndarray (int) with 1 dimension.
                each element specifies relation index of corresponding triple.
        """
        heads_z = self.calc_z(heads)
        tails_z = self.calc_z(tails)
        rel = self.relation[Variable(torch.LongTensor(relations))]

        score = torch.sum(heads_z * rel * tails_z, dim=1)

        return score

    def loss_y(self, score, y):
        return - (score * y - log1exp(score))

    def calc_loss_y(self, heads, relations, tails, y):
        u"""
        [Arguments]
            heads, tails -- ndarray (int) with 2 dimensions.
                0th column contains model indices for each node.
                1st column contains data indices for each node.
            relations -- ndarray (int) with 1 dimension.
                each element specifies relation index of corresponding triple.
            y -- ndarray (int or float) with 1 dimension.
                each element specifies whether corresponding triple holds or not.
                (1 means true, 0 means false)
        """
        score = self.calc_score(heads, relations, tails)
        loss_y = self.loss_y(score, Variable(torch.FloatTensor(y)))

        return loss_y
