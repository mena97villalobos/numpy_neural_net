from nn.op import *
import numpy as np


# implements a log loss layer
class mse_loss_layer(op):

    def __init__(self, i_size, o_size):
        super(mse_loss_layer, self).__init__(i_size, o_size)
        self.grads = np.zeros((o_size, i_size))

    def forward(self, x):
        self.x = x
        self.o = np.dot(x, self.W) + self.b
        return self.o

    # alpha is used as reward in some reinforcement learning envs
    def backward(self, y, rewards=None):
        one_hot = np.zeros(self.o.shape)
        one_hot[np.arange(self.o.shape[0]), y] = 1
        if rewards is not None:
            self.grads = (one_hot - self.o) * rewards
        else:
            self.grads = one_hot - self.o

    def loss(self, y):
        return np.sum((self.o - y) ** 2) / y.size
