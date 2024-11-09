import numpy as np

from simulators import sim_reader
from simulators import simulator

n_train: int = 500000
n_test: int = 5000

sim = simulator()
sim.samples(n_train, n_test)
sim.save()
