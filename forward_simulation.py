import numpy as np

from simulators import sim_reader
from simulators import simulator

sim = simulator()
sim.samples(716800, 307200, 128)
sim.save()
