THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python mlp-relu.py

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mlp.py
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mlp-relu.py
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mlp-relu.py
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python logistic_sgd.py
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python logistic_sgd.py
THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True'
