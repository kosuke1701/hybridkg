#coding: UTF-8

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as _Variable

from log1exp import log1exp

from hybridkg import Variable, Cuda, LVM

class Entity(LVM):
    def __init__(self, n_entity, dim_emb, if_reparam):
        super(Entity, self).__init__(n_entity, if_reparam)

        self.dim_emb = dim_emb

        self.mu = nn.Parameter(
            torch.FloatTensor(np.random.normal(size=(n_entity, dim_emb)))
        )
        if self.if_reparam:
            self.logvar = nn.Parameter(
                torch.FloatTensor(np.ones((n_entity, dim_emb)) * (-10.))
            )

    def _encode(self, idx):
        if self.if_reparam:
            return self.mu[idx], self.logvar[idx]
        else:
            return self.mu[idx]

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
