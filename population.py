import numpy as np
import taichi as ti

from . import activation_funcs
from .data_types import Link, Node, NodeKinds

# Max sizes used to construct efficient lookup tables in reproduction.py
# TODO: I tore out this optimization because it was causing annoying implicit
# cast warnings. Add it back if it seems worthwhile.
# NETWORK_INDEX_DTYPE = ti.uint8
MAX_NETWORK_SIZE = 2**8
MAX_INNOVATIONS = 2**16

@ti.kernel
def copy_one(input_pop: ti.template(), i: int,
             output_pop: ti.template(), o: int):
    for n in range(input_pop.nodes[i].length()):
        output_pop.nodes[o].append(input_pop.nodes[i, n])
    for l in range(input_pop.links[i].length()):
        output_pop.links[o].append(input_pop.links[i, l])

@ti.data_oriented
class Controllers:
    def __init__(self, world_assignments, num_activations, pop, i=None):
        if i is None:
            self.pop = pop
        else:
            self.pop = Population(pop.num_inputs, pop.num_outputs, 1, 1)
            copy_one(pop, i, self.pop, 0)

        self.world_assignments = ti.field(int, shape=self.pop.num_worlds)
        self.world_assignments.from_numpy(world_assignments)
        self.num_activations = num_activations
        self.prev_act = ti.field(
            float,
            shape=(self.pop.num_worlds, num_activations, MAX_NETWORK_SIZE))
        self.curr_act = ti.field(
            float,
            shape=(self.pop.num_worlds, num_activations, MAX_NETWORK_SIZE))

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
        return Controllers(np.zeros(1), self.num_activations, self.pop, i)


@ti.data_oriented
class Population:
    def __init__(self, num_inputs, num_outputs, num_individuals, num_repeats):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_individuals = num_individuals
        self.num_repeats = num_repeats
        self.num_worlds = num_individuals * num_repeats

        node_lists = ti.root.dense(ti.i, self.num_individuals) \
                .dynamic(ti.j, MAX_NETWORK_SIZE, chunk_size=32)
        self.nodes = Node.field()
        node_lists.place(self.nodes)

        link_lists = ti.root.dense(ti.i, self.num_individuals) \
                .dynamic(ti.j, MAX_NETWORK_SIZE, chunk_size=32)
        self.links = Link.field()
        link_lists.place(self.links)

        self.innovation_counter = ti.field(dtype=int, shape=())

    def get_controllers(self, world_assignments, num_activations):
        return Controllers(world_assignments, num_activations, self)

    @ti.kernel
    def clear(self):
        for i in range(self.num_individuals):
            self.nodes[ti.cast(i, int)].deactivate()
            self.links[ti.cast(i, int)].deactivate()

    @ti.func
    def has_room_for(self, i, nodes, links):
        return (self.nodes[i].length() + nodes < MAX_NETWORK_SIZE and
                self.links[i].length() + links < MAX_NETWORK_SIZE)

    @ti.func
    def new_link(self, i, from_node, to_node, weight):
        """Like add_link, but uses a new innovation number."""
        link = Link(from_node, to_node, weight)
        link.innov = ti.atomic_add(self.innovation_counter[None], 1)
        self.links[i].append(link)

    @ti.kernel
    def randomize_all(self):
        for i in range(self.num_individuals):
            self.links[ti.cast(i, int)].deactivate()
            self.nodes[ti.cast(i, int)].deactivate()
            for _ in range(self.num_inputs):
                self.nodes[i].append(
                    Node(NodeKinds.INPUT.value,
                         activation_funcs.random()))
            for _ in range(self.num_outputs):
                self.nodes[i].append(
                    Node(NodeKinds.OUTPUT.value,
                         activation_funcs.random()))

    # This is the activation function for the render* functions, which doesn't
    # use the node activation states, since those would get clobbered when
    # doing many parallel calls for the same individual.
    # TODO: This still has the structure of the recurrent activation function,
    # but it's being used in a non-recurrent way, which isn't right. Add
    # support for non-recurrent networks.
    @ti.func
    def activate_one_r(self, i, inputs):
        result = ti.Vector([0.0] * self.num_outputs)

        # Compute activations for non-input nodes
        for n in range(self.nodes[i].length()):
            if not self.nodes[i, n].deleted:
                raw = 0.0
                for l in range(self.links[i].length()):
                    link = self.links[i, l]
                    if (not link.deleted and link.to_node == n and
                        link.from_node < self.num_inputs):
                        raw += link.weight * inputs[link.from_node]
                if n >= self.num_inputs and n < self.num_inputs + self.num_outputs:
                    result[n - self.num_inputs] = activation_funcs.call(
                        self.nodes[i, n].act_func,
                        raw + self.nodes[i, n].bias)
        return ti.math.clamp(result, 0.0, 1.0)

    @ti.kernel
    def render_one_kernel(self, i: int, output: ti.template()):
        rows, cols = output.shape
        for row, col in ti.ndrange(rows, cols):
            inputs = ti.Vector([row / rows, col / cols])
            # TODO: Handle outputs of varying sizes, not just 1.
            output[row, col] = self.activate_one_r(i, inputs)[0]

    def render_one(self, index, shape):
        result = ti.field(float, shape=shape)
        self.render_one_kernel(index, result)
        return result.to_numpy()

    @ti.kernel
    def render_all_kernel(self, outputs: ti.template()):
        n_i, n_r = self.num_individuals, self.num_repeats
        _, rows, cols = outputs.shape
        # It's important to include at least one of row and col in the outer
        # loop for performance reasons, but both would take too much memory.
        for i, r, row in ti.ndrange(n_i, n_r, rows):
            for col in range(cols):
                w = i * self.num_repeats + r
                inputs = ti.Vector([row / rows, col / cols])
                # TODO: Handle outputs of varying sizes, not just 1.
                outputs[w, row, col] = self.activate_one_r(i, inputs)[0]

    # TODO: Remove? Only here for clearer profile output.
    def render_all(self, outputs):
        self.render_all_kernel(outputs)

def random_population(num_inputs, num_outputs, num_individuals, num_repeats):
    result = Population(num_inputs, num_outputs, num_individuals, num_repeats)
    result.randomize_all()
    return result
