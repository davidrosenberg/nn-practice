#!/usr/bin/python
import tempfile
import os
from sklearn.grid_search import ParameterGrid

configs = []
configs.append({"learning_rate": 0.10, "n_epochs": 5})
configs.append({"learning_rate": 0.15, "n_epochs": 5})
configs.append({"learning_rate": 0.10, "n_epochs": 5})
configs.append({"learning_rate": 0.15, "n_epochs": 5})
configs.append({"learning_rate": 0.20, "n_epochs": 5})
configs.append({"learning_rate": 0.10, "n_epochs": 5})

# Here we'll setup a classic grid search
num_epochs_batchsizeone = 5
grid_params = {}
grid_params["dropout_rate"] = [0, .5]
grid_params["batch_size"] = [1, 10, 100]
configs = list(ParameterGrid(grid_params))
for i,v in enumerate(configs):
    print i, v
    configs[i]["n_epochs"]
    configs[i]["batch_size"]
    num_epochs_batchsizeone


# Set num epochs based on batchsize

use_gpus = False
num_gpus = 6

base_flags = "THEANO_FLAGS=mode=FAST_RUN,floatX=float32"
pythonCmd = "python"
script = "/home/drosen/repos/nn-practice/assignment1/mlp.py"

(f, fname) = tempfile.mkstemp()

gpu_num = 0
for config in configs:
    if not use_gpus:
        flags = base_flags + ",device=cpu"
    elif (num_gpus == 1):
        flags = base_flags + ",device=gpu"
    else:
        flags = base_flags + ",device=gpu" + str(gpu_num)
        gpu_num = (gpu_num + 1) % num_gpus
    confstr = ' '.join("%s=%s" % (key, val)
                       for (key, val) in config.iteritems())
    cmdstr = flags + " " + pythonCmd + " " + script + " with " + confstr + "\n"
    os.write(f, cmdstr)

os.close(f)
cmd = "time parallel -a " + fname
print cmd

# You can then run cmd from the command line, or from Python.
# Makes use of the GNU Parallel program
