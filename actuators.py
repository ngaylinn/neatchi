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

MAX_NETWORK_SIZE = 200

@ti.func
def activate_network(inputs, pop, sp, i, act, b_in, b_out, w, a):
    num_nodes = pop.num_nodes(sp, i)
    num_links = pop.num_links(sp, i)

    # Populate input node activations
    for n in range(pop.num_inputs):
        act[b_in, w, a, n] = inputs[n]

    # Compute activations for non-input nodes
    for n in range(pop.num_inputs, num_nodes):
        node = pop.get_node(sp, i, n)
        raw = 0.0
        for l in range(num_links):
            link = pop.get_link(sp, i, l)
            if link.to_node == n:
                value = act[b_in, w, a, link.from_node]
                raw += value * link.weight
        act[b_out, w, a, n] = activation_funcs.call(
            node.act_func, (node.gain * raw) + node.bias)

    # Copy the output nodes to a vector and return the result.
    outputs = ti.Vector([0.0] * pop.num_outputs)
    for o in range(pop.num_outputs):
        n = num_nodes - pop.num_outputs + o
        outputs[o] = ti.math.clamp(act[b_out, w, a, n], 0.0, 1.0)
    return outputs


@ti.data_oriented
class Actuators:
    """Activate CPPNs from a NeatPopulation."""
    def __init__(self, num_worlds, num_activations, pop):
        self.pop = pop

        # Actuators computes num_activations parallel activations of the CPPNs
        # in pop for each of num_worlds parallel simulations. By default, we
        # assume that there is one world for each individual in the population,
        # but this can be overridden by the caller by setting world_assignments
        # manually.
        self.world_assignments = ti.Vector.field(
            n=2, dtype=int, shape=num_worlds)
        self.world_assignments.from_numpy(
            np.array(list(np.ndindex(pop.population_shape)), dtype=np.int32))

        # Allocate activation buffers for all these parallel CPNNs. A recurrent
        # network needs a double buffer to track the activations from the last
        # time step.
        buffer_count = 1 + pop.is_recurrent
        self.act = ti.field(float, shape=(
            buffer_count, num_worlds, num_activations, MAX_NETWORK_SIZE))

    def reset(self):
        self.act.fill(0.0)

    @ti.func
    def activate(self, inputs, w, a):
        """Activate this CPPN and return its output value(s).

        w is the world_index, indicating which CPPN in the population to use.
        a is the activation index, used to allow multiple parallel activations.
        """
        sp, i = self.world_assignments[w]
        b_in, b_out = 0, 0
        if ti.static(self.pop.is_recurrent):
            b_in = self.pop.buffer_index[None]
            b_out = 1 - b_in
        return activate_network(
            inputs, self.pop, sp, i, self.act, b_in, b_out, w, a)


@ti.data_oriented
class ActivationMaps:
    """Generate maps of activation values for CPPNs from a NeatPopulation."""
    def __init__(self, num_worlds, map_size, pop):
        self.map_size = map_size
        self.pop = pop

        # ActivationMaps renders a map_size x map_size image for every CPPN in
        # pop across num_worlds parallel simulations. By default, we assume
        # there is one world for each individual in the population, but this
        # can be overridden by the caller by setting world_assignments
        # manually.
        self.world_assignments = ti.Vector.field(
            n=2, dtype=int, shape=num_worlds)
        self.world_assignments.from_numpy(
            np.array(list(np.ndindex(pop.population_shape)), dtype=np.int32))

        # Activation buffer for activating all those CPPNs in parallel. We have
        # one for every row of every map.
        num_maps = pop.population_shape[0] * pop.population_shape[1]
        self.act = ti.field(
            float, shape=(1, num_maps, map_size, MAX_NETWORK_SIZE))

        # Rendered maps of activations across 2D space.
        self.maps = ti.field(
            float, shape=pop.population_shape + (map_size, map_size))

    @ti.kernel
    def render(self):
        map_rows = self.pop.population_shape + (self.map_size,)
        for sp, i, r in ti.ndrange(*map_rows):
            for c in range(self.map_size):
                m = sp * i 
                inputs = ti.Vector([r / self.map_size, c / self.map_size])
                # TODO: Support more than one output channel.
                self.maps[sp, i, r, c] = activate_network(
                    inputs, self.pop, sp, i, self.act, 0, 0, m, r)[0]

    @ti.kernel
    def get_one_kernel(self, w: int, output: ti.types.ndarray()):
        sp, i = self.world_assignments[w]
        for r, c, in ti.ndrange(self.map_size, self.map_size):
            output[r, c] = self.maps[sp, i, r, c]

    def get_one(self, w):
        """Return a single rendered ActivationMap for the given world."""
        result = np.zeros((self.map_size, self.map_size))
        self.get_one_kernel(w, result)
        return result

    @ti.func
    def lookup(self, w, r, c):
        """Lookup the activation value for some map location in some world."""
        sp, i = self.world_assignments[w]
        return self.maps[sp, i, r, c]
