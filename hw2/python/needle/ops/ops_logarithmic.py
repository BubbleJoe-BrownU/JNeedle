from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # max_value = array_api.max(Z)
        # self.sum_exp = array_api.sum(array_api.exp(Z - max_value), axis=self.axes)
        # return array_api.log(self.sum_exp) + max_value
        self.max_per_row = array_api.max(Z, axis=self.axes, keepdims=True)
        self.sum_exp = array_api.sum(array_api.exp(Z - self.max_per_row), axis=self.axes)
        return array_api.log(self.sum_exp) + self.max_per_row.squeeze(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, int):
            self.axes = (self.axes, )
        # if self.axes is not specified, default to all dimensions
        if self.axes is None:
            self.axes = [i for i in range(len(node.inputs[0].shape))]
        orig_shape = node.inputs[0].shape
        out_shape = list(out_grad.shape)
        for i in self.axes:
            # when the axes is negative, convert it to positive
            # otherwise when we insert to a negative index, it inserts before it, causing errors
            if i < 0:
                i = list(range(len(node.inputs[0].shape)))[i]
            out_shape.insert(i, 1)
        # print(out_grad)
        # print(node.inputs[0])
        # print(type(self.max_per_row))
        return (out_grad / self.sum_exp).reshape(out_shape) * array_api.exp(node.inputs[0].numpy() - self.max_per_row)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

