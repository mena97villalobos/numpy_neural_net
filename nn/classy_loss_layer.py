from nn.loss_layer import *
import numpy as np


# implements a log loss layer
class classy_loss_layer(loss_layer):

    def __init__(self, i_size, o_size):
        super(classy_loss_layer, self).__init__(i_size, o_size)

    def forward(self, x):
        self.x = x
        self.o = softmax(np.dot(x, self.W) + self.b)
        return self.o

    def loss(self, y):
        one_hot = np.zeros(self.o.shape, dtype=np.int)
        one_hot[np.arange(self.o.shape[0]), y] = 1
        return -np.mean(np.sum(one_hot * np.log(self.o + 1e-15), axis=1))
