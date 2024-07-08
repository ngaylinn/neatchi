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

import taichi as ti

from . import activation_funcs
from .data_types import MAX_NETWORK_SIZE


@ti.data_oriented
class Actuators:
    """Activate CPPNs from a NeatPopulation."""
    def __init__(self, pop, num_activations):
        self.pop = pop

        # Allocate activation buffers for all these parallel CPNNs. A recurrent
        # network needs a double buffer to track the activations from the last
        # time step.
        buffer_count = 1 + pop.is_recurrent
        num_worlds = pop.index.shape[0]
        self.act = ti.field(float, shape=(
            buffer_count, num_worlds, num_activations, MAX_NETWORK_SIZE))

    def reset(self):
        self.act.fill(0.0)

    @ti.func
    def activate(self, inputs, w, a):
        """Activate this CPPN and return its output value(s).

        w is the world index, indicating which parallel simulation this
        activation will be used in.

        a is the activation index, used to allow multiple parallel activations.
        """
        b_in, b_out = 0, 0
        if ti.static(self.pop.is_recurrent):
            b_in = self.pop.buffer_index[None]
            b_out = 1 - b_in

        # Look up which individual in the population is assigned to ativate in
        # this world.
        sp, i = self.pop.index[w]
        num_nodes = self.pop.num_nodes(sp, i)
        num_links = self.pop.num_links(sp, i)

        # Populate input node activations
        for n in range(self.pop.num_inputs):
            self.act[b_in, w, a, n] = inputs[n]

        # Compute activations for non-input nodes
        for n in range(self.pop.num_inputs, num_nodes):
            node = self.pop.get_node(sp, i, n)
            raw = 0.0
            for l in range(num_links):
                link = self.pop.get_link(sp, i, l)
                if link.to_node == n:
                    value = self.act[b_in, w, a, link.from_node]
                    raw += value * link.weight
            self.act[b_out, w, a, n] = activation_funcs.call(
                node.act_func, (node.gain * raw) + node.bias)

        # Copy the output nodes to a vector and return the result.
        outputs = ti.Vector([0.0] * self.pop.num_outputs)
        for o in range(self.pop.num_outputs):
            n = num_nodes - self.pop.num_outputs + o
            outputs[o] = ti.math.clamp(self.act[b_out, w, a, n], 0.0, 1.0)
        return outputs
