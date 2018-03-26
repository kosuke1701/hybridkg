#coding: UTF-8

import torch
from torch.autograd import Variable, Function

from where import where_

#log(1+exp(x))
class Log1Exp(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        x = Variable(x)
        return where_(x>0, torch.log(torch.exp(-x)+1)+x, torch.log(torch.exp(x)+1)).data

    @staticmethod
    def backward(ctx, output_grad):
        x = ctx.saved_variables[0]

        dfdx = torch.sigmoid(x)

        return output_grad * dfdx

log1exp = Log1Exp.apply
