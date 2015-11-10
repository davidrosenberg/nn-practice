#!/usr/bin/python
from sacred import Experiment
ex = Experiment('Logistic Regression practice experiment')


@ex.config
def my_config():
    learning_rate=0.13
    n_epochs=1000,
    dataset='mnist.pkl.gz',
    batch_size=600

@ex.capture
def some_function(a, foo, bar=10):
    print(a, foo, bar)


@ex.automain
def my_main():
    some_function(1, 2, 3)     # 1  2   3
    some_function(1)           # 1  42  'baz'
    some_function(1, bar=12)   # 1  42  12
#    some_function()            # TypeError: missing value for 'a'
#    print "helllo"
