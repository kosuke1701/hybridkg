#coding: UTF-8

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as _Variable

from log1exp import log1exp

from hybridkg import Variable, Cuda, LVM

class PCA_data(LVM):
    def __init__(self, n_entity, dim_latent, dim_data, dim_emb, dim_hidden, with_kg=True):
        super(PCA_data, self).__init__(n_entity, False)

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

            self.f_e_to_x = nn.Linear(dim_emb, dim_latent)

            # self.f_e_to_x = nn.Sequential(
            #     nn.Linear(dim_emb, dim_hidden),
            #     nn.Tanh(),
            #     nn.Linear(dim_hidden, dim_latent)
            # )

            # self.f_e_to_x_h = nn.Sequential(
            #     nn.Linear(dim_emb, dim_hidden),
            #     nn.Tanh()
            # )
            # self.f_e_to_x_mu = nn.Linear(dim_hidden, dim_latent)
            # self.f_e_to_x_logvar = nn.Linear(dim_hidden, dim_latent)
            #
            # self.f_x_to_e_h = nn.Sequential(
            #     nn.Linear(dim_latent, dim_hidden),
            #     nn.Tanh()
            # )
            # self.f_x_to_e_mu = nn.Linear(dim_hidden, dim_emb)
            # self.f_x_to_e_logvar = nn.Linear(dim_hidden, dim_emb)

            self.log_V2 = nn.Parameter(
                torch.FloatTensor(np.ones((1)))
            )

    def _encode(self, idx):
        if not self.with_kg:
            raise Exception("No KG mode.")
        if self.if_reparam:
            raise NotImplementedError("hoge")

        return self.e[idx]

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
        C = self.log_sigma2.exp() * Variable(torch.eye(self.dim_data))
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

            l3 = 0.5 * 1.8378770 * self.dim_latent\
                    + 0.5 * self.log_V2 * self.dim_latent\
                    + 0.5 * torch.sum((mx - x)**2 / self.log_V2.exp(), dim=1)

            # he = self.f_e_to_x_h(self.e[idx])
            # fe_mu = self.f_e_to_x_mu(he)
            # fe_logvar = self.f_e_to_x_logvar(he)
            #
            # le = 0.5 * 1.8378770 * self.dim_latent\
            #         + 0.5 * torch.sum(fe_logvar)\
            #         + 0.5 * torch.sum((fe_mu - x) / fe_logvar.exp(), dim=1)
            #
            # hx = self.f_x_to_e_h(x)
            # fx_mu = self.f_x_to_e_mu(hx)
            # fx_logvar = self.f_x_to_e_logvar(hx)
            #
            # lx = 0.5 * 1.8378770 * self.dim_emb\
            #         + 0.5 * torch.sum(fx_logvar)\
            #         + 0.5 * torch.sum((fx_mu - self.e[idx]) / fx_logvar.exp(), dim=1)
            #
            # l3 = le + lx
        else:
            l3 = 0.

        return l1 + l2 + l3
