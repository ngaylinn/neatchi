import numpy as np
import taichi as ti

from . import activation_funcs
from .population import MAX_NETWORK_SIZE

@ti.real_func
def activate_neuron(pop: ti.template(), i: int, n: int,
                    act: ti.template(), w: int, r: int) -> float:
    # For Renderers, recursive links are ignored. We implement this by
    # refusing to activate a neuron more than once.
    if ti.math.isnan(act[w, r, n]):
        node = pop.nodes[i, n]
        # Set this nodes activation to 0.0 temporarily so that any recurrent
        # links discovered during recursion will get this value instead of
        # looping indefinitely.
        act[w, r, n] = 0.0
        raw = 0.0
        for l in range(pop.links[i].length()):
            link = pop.links[i, l]
            if not link.deleted and link.to_node == n:
                value = activate_neuron(pop, i, link.from_node, act, w, r)
                # TODO: For some reason, weights from input neurons always seem
                # to be 0.0?! But only when I read them here.
                raw += value * 1.0 # link.weight
        act[w, r, n] = activation_funcs.call(node.act_func, raw + node.bias)
    return act[w, r, n]

@ti.func
def activate_network(inputs, pop, i, act, w, r):
    # Initialize activations for all nodes.
    for n in range(pop.nodes[i].length()):
        # Input nodes get activation values taken from the inputs argument.
        if n < pop.num_inputs:
            act[w, r, n] = inputs[n]
        # Other nodes get a nan value to indicate they haven't activated.
        else:
            act[w, r, n] = ti.math.nan

    # Recursively calculate activation values for the output neuron (it
    # comes immediately after the input neurons).
    # TODO: Support more than one channel in the output.
    return ti.math.clamp(
        activate_neuron(pop, i, pop.num_inputs, act, w, r), 0.0, 1.0)

@ti.data_oriented
class NeatRenderers:
    def __init__(self, num_worlds, num_rows):
        self.world_assignments = ti.field(int, shape=num_worlds)
        self.act = ti.field(
            float, shape=(num_worlds, num_rows, MAX_NETWORK_SIZE))

    def update(self, pop, world_assignments):
        self.pop = pop
        self.world_assignments.from_numpy(world_assignments)

    @ti.kernel
    def render_all_kernel(self, pop: ti.template(), outputs: ti.template()):
        num_worlds, rows, cols = outputs.shape
        # For performance reasons, it's important to include at least one of
        # rows or cols in the outer loop. However, putting both in the outer
        # loop requires O(n**2) memory, so just put one there.
        for w, r in ti.ndrange(num_worlds, rows):
            i = self.world_assignments[w]
            for c in range(cols):
                inputs = ti.Vector([r / rows, c / cols])
                outputs[w, r, c] = activate_network(
                    inputs, pop, i, self.act, w, r)

    def render_all(self, outputs):
        self.render_all_kernel(self.pop, outputs)

    @ti.kernel
    def render_one_kernel(self, i: int, outputs: ti.template()):
        rows, cols = outputs.shape
        # This loop structure is parallel to the render_all kernel, except we
        # only render a single world.
        for r in range(rows):
            for c in range(cols):
                inputs = ti.Vector([r / rows, c / cols])
                outputs[r, c] = activate_network(
                    inputs, self.pop, i, self.act, 0, r)

    def render_one(self, index, shape):
        result = ti.field(float, shape=shape)
        self.render_one_kernel(index, result)
        return result.to_numpy()
