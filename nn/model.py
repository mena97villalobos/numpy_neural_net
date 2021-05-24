from nn.funcs import *
from nn.regre_loss_layer import regre_loss_layer
from nn.classy_loss_layer import classy_loss_layer
from nn.dense import dense
import numpy as np


class model():
    def __init__(self, input_size, output_size, hidden_shapes, func_acti, func_acti_grad, loss_func, has_dropout=True,
                 dropout_perc=0.5):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_shapes = hidden_shapes
        self.hidden_amount = len(self.hidden_shapes)
        self.hidden_layers = []
        self.has_dropout = has_dropout
        self.dropout_perc = dropout_perc
        self.populate_layers(func_acti, func_acti_grad, loss_func)

    def populate_layers(self, func_acti, func_acti_grad, loss_func):
        i_size = self.input_size
        for i in range(0, self.hidden_amount):
            self.hidden_layers.append(dense(i_size, self.hidden_shapes[i], func_acti, func_acti_grad))
            i_size = self.hidden_shapes[i] 
        self.loss_layer = self.get_loss_function(i_size, loss_func)

    def get_loss_function(self, i_size, loss_func):
        loss_func_switch = {
            'mse': regre_loss_layer(i_size, self.output_size),
            'softmax': classy_loss_layer(i_size, self.output_size)
        }
        return loss_func_switch[loss_func]

    def forward(self, x, y, train=True):
        self.dropout_masks = []
        data = x
        for i in range(0, self.hidden_amount):
            data = self.hidden_layers[i].forward(data)
            if train and self.has_dropout:  # do dropout only during training
                mask = self.hidden_layers[i].dropout(self.dropout_perc)
                data *= mask
                self.dropout_masks.append(mask)

        o = self.loss_layer.forward(data)
        loss = self.loss_layer.loss(y)
        # print(loss, o)
        return o, loss

    def predict(self, x):
        data = x
        for i in range(0, self.hidden_amount):
            data = self.hidden_layers[i].forward(data)
        o = self.loss_layer.forward(data)
        return o

    # alpha is used for reinforcement learning fir reward
    def backward(self, y, o, rewards=None):
        self.loss_layer.backward(y, rewards)
        prev = self.loss_layer
        for i in reversed(range(self.hidden_amount)):
            self.hidden_layers[i].backward(prev)
            prev = self.hidden_layers[i]
            if self.has_dropout:
                self.hidden_layers[i].grads *= self.dropout_masks[i]  # also mask here

    def update(self, lr):
        for i in range(self.hidden_amount):
            self.hidden_layers[i].update(lr)
        self.loss_layer.update(lr)
