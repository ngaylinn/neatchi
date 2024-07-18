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

        # Since MAX_NETWORK_SIZE is small, we can use thread-local memory to
        # hold the activation buffer. However, for a recurrent network, we need
        # some memory to store the activation values between calls.
        if pop.is_recurrent:
            num_worlds = pop.index.shape[0]
            self.prev_act = ti.field(float,
                shape=(num_worlds, num_activations, MAX_NETWORK_SIZE))

    def reset(self):
        self.prev_act.fill(0.0)

    @ti.func
    def activate_recurrent(self, inputs, w, a):
        act = ti.Vector([0.0] * MAX_NETWORK_SIZE)
        sp, i = self.pop.index[w]
        num_nodes = self.pop.num_nodes(sp, i)
        num_links = self.pop.num_links(sp, i)

        # For a recurrent network, all activation values are taken from the
        # previous time step except for the input nodes, which take on the
        # specified values.
        for n in range(self.pop.num_inputs):
            self.prev_act[w, a, n] = inputs[n]

        # Activate the CPPN. This network is a DAG with nodes sorted by depth,
        # so we can just traverse them in order and compute their final value.
        for n in range(self.pop.num_inputs, num_nodes):
            raw = 0.0
            for l in range(num_links):
                link = self.pop.get_link(sp, i, l)
                if link.to_node == n:
                    value = self.prev_act[w, a, link.from_node]
                    raw += value * link.weight
            node = self.pop.get_node(sp, i, n)
            act[n] = activation_funcs.call(
                node.act_func, (node.gain * raw) + node.bias)

        # Now that activations are computed, put them into memory for the next
        # activation cycle.
        for n in range(self.pop.num_inputs, num_nodes):
            self.prev_act[w, a, n] = act[n]

        # Capture the values from the CPPN's output nodes and return them.
        outputs = ti.Vector([0.0] * self.pop.num_outputs)
        for o in range(self.pop.num_outputs):
            n = num_nodes - self.pop.num_outputs + o
            outputs[o] = ti.math.clamp(act[n], 0.0, 1.0)
        return outputs

    @ti.func
    def activate_forward(self, inputs, w):
        act = ti.Vector([0.0] * MAX_NETWORK_SIZE)
        sp, i = self.pop.index[w]
        num_nodes = self.pop.num_nodes(sp, i)
        num_links = self.pop.num_links(sp, i)

        # Put input values into the activation buffer.
        for n in range(self.pop.num_inputs):
            act[n] = inputs[n]

        # Activate the CPPN. This network is a DAG with nodes sorted by depth,
        # so we can just traverse them in order and compute their final value.
        for n in range(self.pop.num_inputs, num_nodes):
            raw = 0.0
            for l in range(num_links):
                link = self.pop.get_link(sp, i, l)
                if link.to_node == n:
                    value = act[link.from_node]
                    raw += value * link.weight
            node = self.pop.get_node(sp, i, n)
            act[n] = activation_funcs.call(
                node.act_func, (node.gain * raw) + node.bias)

        # Capture the values from the CPPN's output nodes and return them.
        outputs = ti.Vector([0.0] * self.pop.num_outputs)
        for o in range(self.pop.num_outputs):
            n = num_nodes - self.pop.num_outputs + o
            outputs[o] = ti.math.clamp(act[n], 0.0, 1.0)
        return outputs

    def activate(self, inputs, w, a=None):
        if self.pop.is_recurrent:
            self.activate_recurrent(inputs, w, a)
        else:
            self.activate_forward(inputs, w)
