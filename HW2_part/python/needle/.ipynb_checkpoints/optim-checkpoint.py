"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay if weight_decay > 0 else 0.0
        # initialize the parameter momentum table
        for p in self.params:
            self.u[p] = ndl.init.zeros_like(p.data)

    def step(self):
        ### BEGIN YOUR SOLUTION
        # update the parameters with gradients and weight decays
        for p in self.params:
            self.u[p].data = self.momentum*self.u[p].data + (1-self.momentum)*p.grad.data
            p.data = (1-self.lr*self.weight_decay)*p.data - self.lr*self.u[p].data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        return
        for p in self.params:
            # calculate the squared norm of grad
            grad_norm = ndl.power_scalar(ndl.summation(p.grad.data**2), 0.5).data
            if grad_norm <= max_norm:
                continue
            # clip the gradient norm
            p.grad.data /=  grad_norm.data / max_norm
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        print(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay if weight_decay > 0 else 0.0
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        # initialize the running average statistics
        if len(self.m) == 0:
            for p in self.params:
                self.m[p] = ndl.init.zeros_like(p.grad)
                self.v[p] = ndl.init.zeros_like(p.grad)
        # modify running average statistics and update weights
        for p in self.params:
            self.m[p].data = self.beta1 * self.m[p].data + (1-self.beta1) * p.grad.data
            self.v[p].data = self.beta2 * self.v[p].data + (1-self.beta2) * p.grad.data ** 2
            unbiased_m = self.m[p].data / (1 - self.beta1**self.t)
            unbiased_v = self.v[p].data / (1 - self.beta2**self.t)
            p.data = (1-self.lr*self.weight_decay)*p.data - self.lr * unbiased_m / (unbiased_v**0.5 + self.eps)

        ### END YOUR SOLUTION
