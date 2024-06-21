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
                node.act_func, (node.gain * raw) + node.bias)

    # Return a vector of the activation values for just the output nodes.
    return ti.Vector([
        ti.math.clamp(act_out[w, a, n], 0.0, 1.0)
        for n in range(pop.num_inputs, pop.num_inputs + pop.num_outputs)])


@ti.data_oriented
class Actuators:
    """Activate CPPNs from a NeatPopulation."""
    def __init__(self, num_worlds, num_activations, is_recurrent=True):
        self.is_recurrent = is_recurrent
        self.world_assignments = ti.field(int, shape=num_worlds)
        # By default, assume the NeatPopulation has one individual per world
        # and that these correspond 1:1. The caller can override this by
        # passing alternative world assignments to update().
        self.world_assignments.from_numpy(
            np.arange(num_worlds, dtype=np.int32))
        self.act = ti.field(
            float, shape=(num_worlds, num_activations, MAX_NETWORK_SIZE))
        if is_recurrent:
            self.next_act = ti.field(
                float, shape=(num_worlds, num_activations, MAX_NETWORK_SIZE))
        else:
            self.next_act = self.act

    def update(self, world_assignments=None):
        """Update the population CPPNs and world assignments for activation."""
        if world_assignments is not None:
            self.world_assignments.from_numpy(world_assignments)
        # If this is a recurrent network, clear out the activation buffer to
        # make sure state from the last population doesn't leak into this one.
        # For a non-recurrent network, the full conents of this buffer are
        # never read, and always overwritten on each call to activate, so
        # there's no need.
        if self.is_recurrent:
            self.act.fill(0.0)

    @ti.func
    def activate(self, inputs, pop, w, a):
        """Activate this CPPN and return its output value(s).

        w is the world_index, indicating which CPPN in the population to use.
        a is the activation index, used to allow multiple parallel activations.
        """
        result = ti.Vector([0.0] * pop.num_outputs)
        i = self.world_assignments[w]
        result = activate_network(
            inputs, pop, i, self.act, self.next_act, w, a)
        return result

    def finalize_activation(self):
        """Call after all calls to activate() in one time step are done."""
        if ti.static(self.is_recurrent):
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
        # By default, assume the NeatPopulation has one individual per world
        # and that these correspond 1:1. The caller can override this by
        # passing alternative world assignments to update().
        self.world_assignments.from_numpy(
            np.arange(num_worlds, dtype=np.int32))
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

    # TODO: This doesn't work. render_kernel() doesn't see the updated
    # references, so you have to pass pop each time!
    def update(self, pop, world_assignments=None):
        """Update the population CPPNs and world assignments for activation."""
        self.pop = pop
        if world_assignments is not None:
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
    def lookup(self, world_assignments, w, x, y):
        """Lookup the activation value for some map location in some world."""
        i = self.world_assignments[w]
        return self.maps[i, x, y]
