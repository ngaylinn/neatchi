import numpy as np
import taichi as ti

from . import activation_funcs
from .population import MAX_NETWORK_SIZE, NeatPopulation


@ti.data_oriented
class NeatControllers:
    def __init__(self, num_worlds, num_activations):
        self.num_activations = num_activations
        self.world_assignments = ti.field(int, shape=num_worlds)
        self.prev_act = ti.field(
            float,
            shape=(num_worlds, num_activations, MAX_NETWORK_SIZE))
        self.curr_act = ti.field(
            float,
            shape=(num_worlds, num_activations, MAX_NETWORK_SIZE))

    def update(self, pop, world_assignments):
        self.pop = pop
        self.world_assignments.from_numpy(world_assignments)

    @ti.func
    def activate(self, inputs, w, a):
        i = self.world_assignments[w]

        num_nodes = self.pop.nodes[i].length()
        num_links = self.pop.links[i].length()

        # Populate input node activations
        for n in range(self.pop.num_inputs):
            self.prev_act[w, a, n] = inputs[n]

        # Preserve previous activations before computing the next round.
        # TODO: It'd be faster to swap buffer pointers than to copy.
        for n in range(self.pop.num_inputs, num_nodes):
            self.prev_act[w, a, n] = self.curr_act[w, a, n]

        # Compute activations for non-input nodes
        for n in range(self.pop.num_inputs, num_nodes):
            node = self.pop.nodes[i, n]
            if not node.deleted:
                raw = 0.0
                for l in range(num_links):
                    link = self.pop.links[i, l]
                    if not link.deleted and link.to_node == n:
                        value = self.prev_act[w, a, link.from_node]
                        weight = link.weight
                        raw += value * weight
                self.curr_act[w, a, n] = activation_funcs.call(
                    node.act_func, raw + node.bias)

        # Return a vector of the activation values for just the output nodes.
        return ti.Vector([
            ti.math.clamp(self.curr_act[w, a, n], 0.0, 1.0)
            for n in range(self.pop.num_inputs,
                           self.pop.num_inputs + self.pop.num_outputs)])

    def get_one(self, i):
        result = NeatControllers(1, self.num_activations)
        result.update(self.pop.get_one(i), np.array([0], dtype=np.int32))
        return result


