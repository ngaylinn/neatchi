"""The core NEAT algorithm.

This Neat class manages the data allocations used to repeatedly breed a
NeatPopulation of CPPNs. It uses a double buffer to represent the current
generation of CPPNs and a scratch space for generating the next generation. It
also has a Matches object for holding all the metadata needed for a single
breeding process. To actually use any of the CPPNs from the population, use
Actuators or ActivationMaps.
"""

import taichi as ti

from . import population
from . import reproduction


@ti.data_oriented
class Neat:
    """Data allocations for running Neat on a population with given specs."""
    def __init__(self, num_inputs, num_outputs, num_individuals, is_recurrent):
        # TODO: Handle more outputs in the future.
        assert num_outputs == 1
        # Taichi doesn't handle lots of memory allocations / deallocations very
        # well, so allocate all memory we need to breed this population up
        # front, including temp space for breeding.
        self.innovation_counter = ti.field(dtype=int, shape=())
        self.curr_pop = population.NeatPopulation(
            num_inputs, num_outputs, num_individuals, is_recurrent,
            self.innovation_counter)
        self.next_pop = population.NeatPopulation(
            num_inputs, num_outputs, num_individuals, is_recurrent,
            self.innovation_counter)
        self.matches = reproduction.Matches(num_individuals)

    def random_population(self):
        self.curr_pop.clear()
        reproduction.random_init(self.curr_pop)
        #reproduction.validate_all(self.curr_pop)
        return self.curr_pop

    def propagate(self, matches):
        self.matches.update(self.curr_pop, matches)

        self.next_pop.clear()
        reproduction.propagate(self.curr_pop, self.next_pop, self.matches)
        #reproduction.validate_all(self.next_pop)

        # TODO: If you want to update field references inside a Taichi object,
        # the state must be held in Taichi scope, not Python scope. Maybe if
        # you use a single-value scalar field to hold an index into an array of
        # two population fields?
        self.curr_pop, self.next_pop = self.next_pop, self.curr_pop
        return self.curr_pop
