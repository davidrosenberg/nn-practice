import numpy
import theano
import theano.tensor as T
import activations as act



class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, use_bias=True):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            if activation == act.relu1 or activation == act.ReLU:
                # Following He et al. (Delving Deep Into Rectifiers paper)
                print "Initializing a ReLU layer"
                sd = numpy.sqrt(2/float(n_in))
                W_values = numpy.asarray(
                    sd * rng.standard_normal(size=(n_in, n_out)),
                    dtype=theano.config.floatX)
            else:
                print "Initializing non-ReLU layer"
                W_values = numpy.asarray(
                    rng.uniform(
                        low=-numpy.sqrt(6. / (n_in + n_out)),
                        high=numpy.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
                )
                if activation == theano.tensor.nnet.sigmoid:
                    W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W

        if not use_bias:
            self.params = [self.W]
            lin_output = T.dot(input, self.W)
        else:
            if b is None:
                b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
                b = theano.shared(value=b_values, name='b', borrow=True)
            self.b = b
            self.params = [self.W, self.b]
            lin_output = T.dot(input, self.W) + self.b

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
