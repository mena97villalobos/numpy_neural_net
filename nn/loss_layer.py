from nn.op import *
import numpy as np


# implements a log loss layer
class loss_layer(op):

    def __init__(self, i_size, o_size):
        super(loss_layer, self).__init__(i_size, o_size)
        self.grads = np.zeros((o_size, i_size))

    # alpha is used as reward in some reinforcement learning envs
    def backward(self, y, rewards=None):
        one_hot = np.zeros(self.o.shape)
        one_hot[np.arange(self.o.shape[0]), y] = 1
        if rewards is not None:
            self.grads = (one_hot - self.o) * rewards
        else:
            self.grads = one_hot - self.o

    def loss(self, y):
        pass
