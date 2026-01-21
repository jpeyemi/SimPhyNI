# Expose the main classes or modules from the package
from .Simulation.tree_simulator import TreeSimulator
from .Simulation.pair_statistics import pair_statistics
from .Simulation.simulation import (simulate_glrates_bit, sim_bit, compres)
# Add here for simulation methods if we want to allow running within python

__all__ = ["TreeSimulator","pair_statistics", "simulate_glrates_bit",
    "sim_bit",
    "compres",]
