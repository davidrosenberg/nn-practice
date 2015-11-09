# -*- coding: utf-8 -*-
"""
Theano tutorial from deeplearning.net

Created on Wed Oct 14 16:47:05 2015

@author: drosen
"""

import theano.tensor as T
from theano import function q
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x,y],z)

from theano import pp
print pp(z)

z.eval({x:6.4, y:12.1})

## Adding two matrices
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f= function([x,y],z)

f([[1,2],[3,4]], [[10,20],[30,40]])  # output is a numpy array

import numpy
f(numpy.array([[1,2],[3,4]]), [[10,20],[30,40]])  # output is a numpy array

## Exercise
import theano
a = theano.tensor.vector()  #declare variable
out = a * a ** 10 #build symbolic expression
f = theano.function([a], out) #compile function
print(f([0,1,2]))

## Make something to compute a^2 + b^2 +2 a b  for parallel vectors
a = theano.tensor.vector()
b = theano.tensor.vector()
out = a * a + b * b + 2 * a * b
f = theano.function([a,b],out)
print(f([0,1,2],[1,2,3]))

## Logistic
x = T.dmatrix("x")
s = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s)
logistic([[0,1], [-1,-2]]) # elementwise because constituent operations are elementwise

## Equivalently, can use tanh function
s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = function([x], s2)
logistic2([[0,1], [-1,-2]]) # elementwise because constituent operations are elementwise

## Function can also have multiple outputs
a, b = T.dmatrices('a', 'b')
c, d = T.dmatrices('c', 'd')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff ** 2
f = theano.function([a, b], [diff, abs_diff, diff_squared])
f([[1,1], [1,1]], [[0,1],[2,3]])

## Default value for argument
x, y = T.dscalars('x', 'y')
z = x + y
f = theano.function([x, theano.Param(y, default = 1)], z)
f(33)
f(33,2)

from theano import function
from theano import Param
x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
## for Param(y, default=1), we can set by name with y=0
## For Param(w, name='w_by_name'), we can set by name with w_by_name=3
f = function([x, Param(y, default=1), Param(w, default=2, name='w_by_name')], z) 
f(33)
f(33, 2)
f(33, 0, 1)
f(33, w_by_name=1)
f(33, w_by_name=1, y=0)

## Shared variables
from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])

## shared(0) is a shared value initialized to zero.  Can check its value as
state.get_value()
state.set_value(-1)

## called shared because its value is teh same in all functions that
## use it The updates parameter to theano.function gets a list of
## pairs of the form (shared-variable, new expression).  Can also be a
## dictionary whose keys are shared-variables and values ar ethe new
## expressions.

accumulator(1)
state.get_value()

decrementor = function([inc], state, updates=[(state, state-inc)])

decrementor(1)
state.get_value()

## If we expressed a formula using a shared variable, but we don't
## want to use its value, we can use the "givens" parameter of
## function, which replaces a particular node in a graph for the
## purpose of one particular function.
fn_of_state = state * 2 + inc
## type of foo must match the shared variable we are replacing with the "givens"
foo = T.scalar(dtype = state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state,foo)])
state.get_value()
skip_shared(1,3)

## givens can be used to replace any symbolic variable, not just a
## share varible; can replace constatns, expressions, in general;
## CAREFUL; substitutions must be able to work in any order

## in practice, can think about givens as a mecahnism that allows you
## to replace any part of your formula with a differenet expression
## that evaluates to a tensor of same shape and dtype

## Best way to think about random nubmers in Thenao is as a random
## variable... or a random stream; think of it as a shared variable

from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True) # not updating rv_n.rng (internal state of random number generator is not affected by call)
nearly_zeros = function([], rv_u+rv_u -2 * rv_u)  # a single random variable is drawn at most once during a single function execution, which is why all these rv_u 's are the same and the result is zero

## Seeding streams



--------------
...
-------------
##http://deeplearning.net/software/theano/tutorial/aliasing.html
## Memory Aliasing
#Theano stuff is stored in asedparate memory space from other python variables
Borrowing when Creating Shared Variables
A borrow argument can be provided to the shared-variable constructor.

import numpy, theano
np_array = numpy.ones(2, dtype='float32')

s_default = theano.shared(np_array)
s_false   = theano.shared(np_array, borrow=False)
s_true    = theano.shared(np_array, borrow=True)
By default (s_default) and when explicitly setting borrow=False, the shared variable we construct gets a [deep] copy of np_array. So changes we subsequently make to np_array have no effect on our shared variable.

np_array += 1 # now it is an array of 2.0 s

print(s_default.get_value())
print(s_false.get_value())
print(s_true.get_value())

[ 1.  1.]
[ 1.  1.]
[ 2.  2.]

s_true is using the np_array object as it’s internal buffer

However, this aliasing of np_array and s_true is not guaranteed to occur, and may occur only temporarily even if it occurs at all. It is not guaranteed to occur because if Theano is using a GPU device, then the borrow flag has no effect.

Take home message:

It is a safe practice (and a good idea) to use borrow=True in a shared variable constructor when the shared variable stands for a large object (in terms of memory footprint) and you do not want to create copies of it in memory.

It is not a reliable technique to use borrow=True to modify shared variables through side-effect, because with some devices (e.g. GPU devices) this technique will not work.

### Borrowing when Accessing Value of Shared Variables
A borrow argument can also be used to control how a shared variable’s value is retrieved.

s = theano.shared(np_array)
v_false = s.get_value(borrow=False) # N.B. borrow default is False
v_true = s.get_value(borrow=True)

It is safe (and sometimes much faster) to use get_value(borrow=True) when your code does not modify the return value. Do not use this to modify a ``shared`` variable by side-effect because it will make your code device-dependent. Modification of GPU variables through this sort of side-effect is impossible.

### Assigning (don't really understand this)
A standard pattern for manually updating the value of a shared variable is as follows:

s.set_value(
    some_inplace_fn(s.get_value(borrow=True)),
    borrow=True)

## Borrow for constucting function objects
f2 = function([],
              Out(sandbox.cuda.basic_ops.gpu_from_host(tensor.exp(x)),
                  borrow=True))

When an input x to a function is not needed after the function returns and you would like to make it available to Theano as additional workspace, then consider marking it with In(x, borrow=True). It may make the function faster and reduce its memory requirement. When a return value y is large (in terms of memory footprint), and you only need to read from it once, right away when it’s returned, then consider marking it with an Out(y, borrow=True).
