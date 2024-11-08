import numpy as np

from simulators import sim_reader
from simulators import simulator

n_train: int = 500000
n_val: int = 100000
n_test: int = 20000

sim = simulator()
sim.samples(n_train, n_val, n_test)
sim.save()
