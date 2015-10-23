# -*- coding: utf-8 -*-
"""
From http://stackoverflow.com/questions/26497564/theano-hiddenlayer-activation-function
Created on Fri Oct 23 14:24:17 2015

@author: drosen
"""

def relu1(x):
    return T.switch(x<0, 0, x)

def relu2(x):
    return T.maximum(x, 0)

def relu3(x):
    return x * (x > 0)


z = numpy.random.normal(size=[1000, 1000])
for f in [relu1, relu2, relu3]:
    x = theano.tensor.matrix()
    fun = theano.function([x], f(x))
#    %timeit fun(z)
    assert numpy.all(fun(z) == numpy.where(z > 0, z, 0))

Output: (time to compute ReLU function)
>100 loops, best of 3: 3.09 ms per loop
>100 loops, best of 3: 8.47 ms per loop
>100 loops, best of 3: 7.87 ms per loop

for f in [relu1, relu2, relu3]:
    x = theano.tensor.matrix()
    fun = theano.function([x], theano.grad(T.sum(f(x)), x))
    %timeit fun(z)
    assert numpy.all(fun(z) == (z > 0)