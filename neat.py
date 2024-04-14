import numpy as np
import taichi as ti

from . import population
from . import reproduction

@ti.data_oriented
class Neat:
    """Data allocations for running Neat on a population with given specs."""
    def __init__(self, num_inputs, num_outputs, num_individuals, num_repeats):
        # Taichi doesn't handle lots of memory allocations / deallocations very
        # well, so allocate all memory we need to breed this population up
        # front, including temp space for breeding.
        self.__curr_pop = population.Population(
            num_inputs, num_outputs, num_individuals, num_repeats)
        self.__next_pop = population.Population(
            num_inputs, num_outputs, num_individuals, num_repeats)
        self.matches = reproduction.Matches(num_individuals)

    def random_population(self):
        self.__curr_pop.clear()
        self.__curr_pop.randomize_all()
        return self.__curr_pop

    def propagate(self, parent_selections, mate_selections):
        self.__next_pop.clear()
        self.matches.update(
            self.__curr_pop, parent_selections, mate_selections)
        reproduction.propagate(self.__curr_pop, self.__next_pop, self.matches)
        self.__curr_pop, self.__next_pop = self.__next_pop, self.__curr_pop
        return self.__curr_pop
