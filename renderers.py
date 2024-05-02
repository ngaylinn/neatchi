import taichi as ti

from . import activation_funcs
from .population import MAX_NETWORK_SIZE


@ti.func
def activate_network(inputs, pop, i, act, w, r):
    num_nodes = pop.nodes[i].length()
    num_links = pop.links[i].length()

    for n in range(pop.num_inputs):
        act[w, r, n] = inputs[n]
    for n in range(pop.num_inputs, num_nodes):
        node = pop.nodes[i, n]
        if not node.deleted:
            raw = 0.0
            for l in range(num_links):
                link = pop.links[i, l]
                if not link.deleted and link.to_node == n:
                    value = act[w, r, link.from_node]
                    raw += value * link.weight
                    act[w, r, n] = activation_funcs.call(
                        node.act_func, raw + node.bias)
    # TODO: Support more than one output channel.
    return ti.math.clamp(act[w, r, num_nodes - 1], 0.0, 1.0)


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
        # TODO: Fill in the ndarray directly?
        result = ti.field(float, shape=shape)
        self.render_one_kernel(index, result)
        return result.to_numpy()

