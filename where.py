#coding: UTF-8

import torch
from torch.autograd import Variable, Function

class Where(Function):
    @staticmethod
    def forward(ctx, condition, _if, _else):
        ctx.save_for_backward(condition, _if, _else)

        _if = _if.clone().masked_fill_(condition!=1, 0)
        _else = _else.clone().masked_fill_(condition, 0)

        return _if + _else


    @staticmethod
    def backward(ctx, grad_output):
        condition, _if, _else = ctx.saved_variables

        grad_condition = None

        grad_if = Variable(grad_output.data.clone().masked_fill_(condition.data!=1, 0))
        grad_else = Variable(grad_output.data.clone().masked_fill_(condition.data, 0))

        return grad_condition, grad_if, grad_else

where_ = Where().apply
