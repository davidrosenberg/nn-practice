import numpy
import theano
import theano.tensor as T
import activations as act


def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)


def relu1(x):
    return T.switch(x < 0, 0, x)


def get_activation(name):
    mapping = {
        "relu": ReLU,
        "tanh": T.tanh,
        "sigmoid": T.nnet.sigmoid
    }
    return mapping.get(name)
