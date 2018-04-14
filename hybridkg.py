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

class LVM(nn.Module):
    u"""
    Abstract class for LVM.

    [Note]
        1. If self.if_reparam is True, embeddings of entities in this LVM are sampled
           using reparameterization trick.
           (It assumes that posterior of embeddings is modeled by Gaussian and mean field approximation.)

           If self.if_reparam is False, embeddings of entities in this LVM will be
           point estimated.

        2. Class function _encode(self, idx) must be implemented in its child classes.
           This function will return embeddings of entities which are specified by idx.
           (idx is 1 dimensional torch.LongTensor.)

           If self.if_reparam is True, _encode(self, idx) must return tuple of 2 Variables.
           First Variable is mean vecotrs for entities specified by idx.
           (Each row corresponds to each entry in idx.)
           Second Variable is logarithm of diagonal variance vectors for entities specified by idx.
           (Each row corresponds to each entry in idx.)

           If self.if_reparam is False, _encode(self, idx) must return one Variable.
           It is embedding vectors of entities specified by idx.
           (Each row corresponds to each entry in idx.)

           In both cases, returned Variable's tensor must be 2 dimensional torch.FloatTensor,
           and their shapes are (dimension-of-idx, dimension-of-embedding).
    """
    def __init__(self, n_entity, if_reparam):
        super(LVM, self).__init__()

        self.if_reparam = if_reparam
        self.n_entity = n_entity

    def encode(self, idx):
        if not isinstance(idx, _Variable):
            idx = Variable(torch.LongTensor(idx))

        return self._encode(idx)

    def _encode(self, idx):
        raise NotImplementedError("")

    def get_n_entity(self):
        return self.n_entity

class HybridDistMult(nn.Module):
    u"""
    This class calculates loss for knowledge graph, i.e. - log[p(triples|entity embeddings)].

    p(triples|entity embeddings) =
                              | sigmoid(e_h * M_r * e_t)      (if (h,r,t) is true)
        \prod_{(h,r,t) \in T} |
                              | 1 - sigmoid(e_h * M_r * e_t)  (if (h,r,t) is false)

    [Note]
        1. LVMs in a model must be listed in its self.models.

        2. Each entitiy (or node) is represented by 2 types of index (index of LVM model
           and index of the entity in the LVM).

        3. Use function calc_loss_y(self, heads, relations, tails, y) to calculate loss for mini-batch
           of triples.
    """
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

        [Return values]
            loss_y
            loss_y -- Calculated loss -logp(triple|embedding) for triples.
                      Variable of 1 dimensional torch.FloatTensor.
                      Each entry corresponds to each triple in given mini-batch of triples.
        """
        score = self.calc_score(heads, relations, tails)
        loss_y = self.loss_y(score, Variable(torch.FloatTensor(y)))

        return loss_y
