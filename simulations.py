import numpy as np

from simulators import reader
from simulators import simulator

sim = simulator()
sim.samples(1024000)
sim.save()
