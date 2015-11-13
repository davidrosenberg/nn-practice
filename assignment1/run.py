#!/usr/bin/python
import tempfile
import os

configs = []
configs.append({"learning_rate": 0.10, "n_epochs": 10})
configs.append({"learning_rate": 0.15, "n_epochs": 10})
configs.append({"learning_rate": 0.20, "n_epochs": 10})
configs.append({"learning_rate": 0.10, "n_epochs": 15})

num_gpus = 6

base_flags = "THEANO_FLAGS=mode=FAST_RUN,floatX=float32"
pythonCmd = "python"
script = "/home/drosen/repos/nn-practice/assignment1/mlp.py"

(f, fname) = tempfile.mkstemp()

gpu_num = 0
for config in configs:
    if (num_gpus == 1):
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
