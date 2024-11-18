import numpy as np

from simulators import sim_reader
from simulators import simulator

sim = simulator()
sim.samples(102400, 10)
sim.save()
