import numpy as np

from simulators import reader
from simulators import simulator

sim = simulator()
sim.samples(1000000)
sim.save()
