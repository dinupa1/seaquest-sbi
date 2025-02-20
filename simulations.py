import numpy as np

from simulators import reader
from simulators import simulator

sim = simulator()
sim.samples(600000, 200000, 200000)
sim.save()
