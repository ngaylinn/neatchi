"""Classes for using a CPPN to process inputs or render an image.

The Actuators class is used to call CPPNs from a NeatPopulation from a number
of parallel worlds, possibly with multiple parallel activations in each world.
This requires allocating memory proportional to MAX_NETWORK_SIZE * num_worlds *
num_activations. Mappings from population indices to world indices are handled
with the world_assignments field, which get updated each time the population is
propagated.

The ActivationMap class is similar, except its used to render a 2D map of
activation values for each CPPN, which then can be referenced from many
parallel worlds. Whereas Actuators.activate() computes the CPPN's output value
on demand from a given input, ActivationMap.lookup() only fetches a cached
value that was computed in ActivationMap.update().
"""

import numpy as np
import taichi as ti

from . import activation_funcs
from .population import MAX_NETWORK_SIZE

@ti.func
def activate_network(inputs, pop, i, act_in, act_out, w, a):
    num_nodes = pop.nodes[i].length()
    num_links = pop.links[i].length()

    # Populate input node activations
    for n in range(pop.num_inputs):
        act_in[w, a, n] = inputs[n]

    # Compute activations for non-input nodes
    for n in range(pop.num_inputs, num_nodes):
        node = pop.nodes[i, n]
        if not node.deleted:
            raw = 0.0
            for l in range(num_links):
                link = pop.links[i, l]
                if not link.deleted and link.to_node == n:
                    value = act_in[w, a, link.from_node]
                    raw += value * link.weight
            act_out[w, a, n] = activation_funcs.call(
                node.act_func, raw + node.bias)

    # Return a vector of the activation values for just the output nodes.
    return ti.Vector([
        ti.math.clamp(act_out[w, a, n], 0.0, 1.0)
        for n in range(pop.num_inputs, pop.num_inputs + pop.num_outputs)])


@ti.data_oriented
class Actuators:
    """Activate CPPNs from a NeatPopulation."""
    def __init__(self, num_worlds, num_activations, recurrent=True):
        self.recurrent = recurrent
        self.world_assignments = ti.field(int, shape=num_worlds)
        self.act = ti.field(
            float, shape=(num_worlds, num_activations, MAX_NETWORK_SIZE))
        if recurrent:
            self.next_act = ti.field(
                float, shape=(num_worlds, num_activations, MAX_NETWORK_SIZE))
        else:
            self.next_act = self.act

    def update(self, pop, world_assignments):
        """Update the population CPPNs and world assignments for activation."""
        self.pop = pop
        self.world_assignments.from_numpy(world_assignments)
        self.act.fill(0.0)

    @ti.func
    def activate(self, inputs, w, a):
        """Activate this CPPN and return its output value(s).

        w is the world_index, indicating which CPPN in the population to use.
        a is the activation index, used to allow multiple parallel activations.
        """
        result = ti.Vector([0.0] * self.pop.num_outputs)
        i = self.world_assignments[w]
        result = activate_network(
            inputs, self.pop, i, self.act, self.next_act, w, a)
        return result

    def finalize_activation(self):
        """Call after all calls to activate() in one time step are done."""
        if ti.static(self.recurrent):
            # Once all the activations have been computed, swap the activation
            # buffers so that the last set of outputs serves as inputs in the
            # next activation.
            self.act, self.next_act = self.next_act, self.act


@ti.data_oriented
class ActivationMaps:
    """Generate maps of activation values for CPPNs from a NeatPopulation."""
    def __init__(self, num_worlds, num_individuals, map_size):
        self.map_size = map_size
        self.world_assignments = ti.field(int, shape=num_worlds)
        self.act = ti.field(
            float, shape=(num_individuals, map_size, MAX_NETWORK_SIZE))
        self.maps = ti.field(
            float, shape=(num_individuals, map_size, map_size))

    @ti.kernel
    def render_kernel(self):
        for i, r in ti.ndrange(self.pop.num_individuals, self.map_size):
            for c in range(self.map_size):
                inputs = ti.Vector([r / self.map_size, c / self.map_size])
                # TODO: Support more than one output channel.
                self.maps[i, r, c] = activate_network(
                    inputs, self.pop, i, self.act, self.act, i, r)[0]

    def update(self, pop, world_assignments):
        """Update the population CPPNs and world assignments for activation."""
        self.pop = pop
        self.world_assignments.from_numpy(world_assignments)
        self.render_kernel()

    @ti.kernel
    def get_one_kernel(self, i: int, output: ti.types.ndarray()):
        for r, c, in ti.ndrange(self.map_size, self.map_size):
            output[r, c] = self.maps[i, r, c]

    def get_one(self, w):
        """Return a single rendered ActivationMap for the given world."""
        result = np.zeros((self.map_size, self.map_size))
        self.get_one_kernel(self.world_assignments[w], result)
        return result

    @ti.func
    def lookup(self, w, x, y):
        """Lookup the activation value for some map location in some world."""
        i = self.world_assignments[w]
        return self.maps[i, x, y]
