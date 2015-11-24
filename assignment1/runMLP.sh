THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python mlp-relu.py

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mlp.py
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mlp-relu.py
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mlp-relu.py
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python logistic_sgd.py
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python logistic_sgd.py
THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' python mlp.py
THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' python mlp.py with n_epochs=20 batch_size=100
THEANO_FLAGS='floatX=float32,device=cpu,nvcc.fastmath=True' python mlp.py with n_epochs=20 batch_size=100
