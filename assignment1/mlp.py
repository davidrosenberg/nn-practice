#!/usr/bin/python
from sacred import Experiment
ex = Experiment('Multilayer Perceptron practice experiment')

from sacred.observers import MongoObserver

import os
import timeit

import numpy

import theano
import theano.tensor as T


import data_loader
from logistic_layer import LogisticRegression
from hidden_layer import HiddenLayer, DropoutHiddenLayer, _dropout_from_layer
# import misc
# from theano.compile.sharedvalue import SharedVariable, shared

import activations as act


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, activations="tanh",
                 use_bias=True, dropout=False, dropout_rate=0):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        #
        # For Dropout, we basically need to set up two different MLPs
        # - one with dropout layers (used for training) and one for
        # prediction. [Question -- can we get error bounds if we run
        # forward propagation on the random dropout network a bunch of
        # times?  Might these be calibrated probabilities? Probably not...

        # Not sure if this is necessary -- but just in case for now.
        if not dropout:
            dropout_rate = 0

        activation = act.get_activation(activations)

        next_layer_input = input
        next_dropout_layer_input = _dropout_from_layer(
            rng, input, dropout_rate=dropout_rate)

        next_dropout_layer = DropoutHiddenLayer(
            rng=rng, input=next_dropout_layer_input,
            n_in=n_in, n_out=n_hidden, activation=activation,
            use_bias=use_bias, dropout_rate=dropout_rate)

        next_dropout_layer_input = next_dropout_layer.output

        # Reuse the parameters from the dropout layer here, in a different
        # path through the graph.
        # [Could be a constructor that takes a dropout hidden layer.]
        next_layer = HiddenLayer(
            rng=rng, input=next_layer_input,
            activation=activation,
            # scale the weight matrix W with probability of keeping
            W=next_dropout_layer.W * (1 - dropout_rate),
            b=next_dropout_layer.b,
            n_in=n_in, n_out=n_hidden,
            use_bias=use_bias)

        next_layer_input = next_layer.output

        # Now we set up the logistic regression (i.e. softmax) output
        # layers for the dropout network and the regular network
        self.dropout_output_layer = LogisticRegression(
            input=next_dropout_layer_input, n_in=n_hidden, n_out=n_out)

        self.output_layer = LogisticRegression(
            input=next_layer_input, n_in=n_hidden, n_out=n_out,
            W=self.dropout_output_layer.W * (1-dropout_rate),
            b=self.dropout_output_layer.b)

        # self.L1 = (abs(self.hiddenLayer.W).sum()
        #            + abs(self.logRegressionLayer.W).sum())
        # self.L2_sqr = ((self.hiddenLayer.W ** 2).sum()
        #                + (self.logRegressionLayer.W ** 2).sum())

        self.dropout_nll = self.dropout_output_layer.negative_log_likelihood
        self.dropout_errors = self.dropout_output_layer.errors
        self.nll = self.output_layer.negative_log_likelihood
        self.errors = self.output_layer.errors

        # The parameters for dropout and non-dropout are the same, but
        # we need to add the ones in the dropout layers, because those
        # are the shared variables... the ones in next_layer are
        # derived versions.
        self.params = self.dropout_output_layer.params + next_dropout_layer.params

        # keep track of model input
        self.input = input


@ex.capture
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500,
             activation="tanh", dropout=False, dropout_rate=0.5,
             use_bias=True,
             rng=numpy.random.RandomState(1234)):

    datasets = data_loader.load_data(dataset)

    train_set_x, train_set_y = datasets["data"][0]
    valid_set_x, valid_set_y = datasets["data"][1]
    test_set_x, test_set_y = datasets["data"][2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=datasets["info"]["xdim"],
                     n_hidden=n_hidden,
                     n_out=datasets["info"]["y_num_categories"],
                     activations=activation,
                     use_bias=use_bias,
                     dropout=dropout,
                     dropout_rate=dropout_rate)

    cost = (
        classifier.nll(y)
        # + L1_reg * classifier.L1
        # + L2_reg * classifier.L2_sqr
    )
    dropout_cost = classifier.dropout_nll(y)
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (stored in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []

    print "\nNow computing partial derivative w.r.t. each parameter:"

    for param in classifier.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)

    # For use in theano.function, updates is a list,tuple,or dict
    # that's iterable over pairs (shared_variable, new_expression)
    updates = [
        (param, param - learning_rate * gp)
        for param, gp in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # at the same time updates the parameters of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    # look at this many examples regardless
    patience = 10000  
    # wait this much longer when a new best is found
    patience_increase = 2
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995
    # go thru this many minibatches before checking the network on
    # the validation set (or once per epoch, whichever is smaller)
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Store experiment information
    ex.info["run_time"] = end_time - start_time
    ex.info["validation_perf"] = best_validation_loss * 100
    ex.info["num_epochs"] = epoch
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))


@ex.config
def my_config():
    learning_rate = 0.01
    n_epochs = 25
    data_path = "/home/drosen/repos/DeepLearningTutorials/data"
    datasetname = 'mnist.small.pkl.gz'
    #datasetname = 'mnist.pkl.gz'
    dataset = os.path.join(data_path, datasetname)
    batch_size = 100
    n_hidden = 500
    theano_flags = "mode=FAST_RUN,device=gpu,floatX=float32"
    os.environ["THEANO_FLAGS"] = theano_flags
    db_name = "MY_DB"
    ex.observers.append(MongoObserver.create(db_name=db_name))
    random_seed = 1234
    rng = numpy.random.RandomState(random_seed)
    activation = "tanh"
    dropout=True
    dropout_rate=0.5
    use_bias=False

@ex.automain
def my_main():
    test_mlp()
