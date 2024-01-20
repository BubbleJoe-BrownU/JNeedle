"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

import needle as ndl


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features, fan_out=out_features, device=device, dtype=dtype))
        if self.has_bias:
            self.bias = Parameter(ndl.transpose(init.kaiming_uniform(fan_in=out_features,fan_out=1, device=device, dtype=dtype)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.has_bias:
            # sometimes the input has extra dims, though only very occasionally
            # so we broadcast the bias to X.shape[:-1] + [out_features] to cope with this scenario
            return X @ self.weight + ndl.broadcast_to(self.bias, list(X.shape[:-1]) + [self.out_features])
        return X @ self.weight
        ### END YOUR SOLUTION


class Flatten(Module):
    """
    Takes in a tensor of shape `(B,X_0,X_1,...)`, and flattens all non-batch dimensions so that the output is of shape `(B, X_0 * X_1 * ...)`
    """
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        return X.reshape((batch_size, -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ndl.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if len(self.modules) == 0:
            return x
        for m in self.modules:
            x = m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # max is non-differentiable, so we use detached max values of logits
        normalized_logits = logits - Tensor(np.max(logits.numpy(), axis=-1)).reshape((-1, 1)).data
        probs = ndl.exp(normalized_logits) / ndl.broadcast_to(ndl.summation(ndl.exp(normalized_logits), axes=-1).reshape((-1, 1)), logits.shape)
        # num_cls = max(len(set(y.numpy())), max(y.numpy())+1)
        y_onehot = init.one_hot(logits.shape[1], y)
        loss = ndl.summation(- Tensor(y_onehot) * ndl.log(probs)).reshape(1)/ logits.shape[0]
        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # if the batch mean and var is reshaped to (1, -1), then it will also implicitly broadcast the running mean and var
            mean = ndl.summation(x, axes=0) / x.shape[0]
            var = ndl.summation((x - ndl.broadcast_to(mean, x.shape))**2, axes=0) / x.shape[0]
            # modify the running average statistics with detached mean and vars to avoid extending computation graphs
            self.running_mean.data = (1-self.momentum)*self.running_mean.data + self.momentum*mean.data
            self.running_var.data = (1-self.momentum)*self.running_var.data + self.momentum*var.data
            return ndl.broadcast_to(self.weight, x.shape) * (x - ndl.broadcast_to(mean, x.shape)) / ndl.power_scalar(ndl.broadcast_to(var, x.shape) + self.eps, 0.5) + ndl.broadcast_to(self.bias, x.shape)

        return ndl.broadcast_to(self.weight, x.shape) * (x - ndl.broadcast_to(self.running_mean, x.shape)) / ndl.power_scalar(ndl.broadcast_to(self.running_var, x.shape) + self.eps, 0.5) + ndl.broadcast_to(self.bias, x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.w = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.b = Parameter(init.zeros(dim, device=device, dtype=dtype))

        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # if self.training:
        mean = ndl.summation(x, axes=-1).reshape((-1, 1)) / x.shape[1]
        var = ndl.summation((x - ndl.broadcast_to(mean, x.shape))**2, axes=-1).reshape((-1, 1)) / x.shape[1]
        return ndl.broadcast_to(self.w, x.shape) * (x - ndl.broadcast_to(mean, x.shape)) / ndl.power_scalar(ndl.broadcast_to(var+self.eps, x.shape), 0.5) + ndl.broadcast_to(self.b, x.shape)
            
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = Tensor(np.random.binomial(1, 1-self.p, size=x.shape).astype(x.dtype))
            return x * mask / (1-self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
