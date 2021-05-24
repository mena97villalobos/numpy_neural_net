from nn.loss_layer import *
import numpy as np


# implements a log loss layer
class regre_loss_layer(loss_layer):

    def __init__(self, i_size, o_size):
        super(regre_loss_layer, self).__init__(i_size, o_size)

    def forward(self, x):
        self.x = x
        self.o = np.dot(x, self.W) + self.b
        return self.o

    def backward(self, y, rewards=None):
        print(self.o, y)
        if rewards is not None:
            self.grads = self.o * rewards
        else:
            self.grads = self.o

    def loss(self, y):
        return np.sum((self.o - y) ** 2) / y.size
