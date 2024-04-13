import numpy as np
import taichi as ti

from . import activation_funcs
from . import reproduction
from .data_types import Link, Node, NodeKinds

# Max sizes used to construct efficient lookup tables in reproduction.py
# TODO: I tore out this optimization because it was causing annoying implicit
# cast warnings. Add it back if it seems worthwhile.
# NETWORK_INDEX_DTYPE = ti.uint8
MAX_NETWORK_SIZE = 2**8
MAX_INNOVATIONS = 2**16


@ti.data_oriented
class Population:
    def __init__(self, num_inputs, num_outputs, num_individuals, num_repeats):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_individuals = num_individuals
        self.num_repeats = num_repeats
        self.num_worlds = num_individuals * num_repeats

        # TODO: Tune sizes?
        node_lists = ti.root.dense(ti.i, self.num_individuals) \
                .dynamic(ti.j, MAX_NETWORK_SIZE, chunk_size=32)
        self.nodes = Node.field()
        node_lists.place(self.nodes)

        # TODO: Tune sizes?
        link_lists = ti.root.dense(ti.i, self.num_individuals) \
                .dynamic(ti.j, MAX_NETWORK_SIZE, chunk_size=32)
        self.links = Link.field()
        link_lists.place(self.links)

        self.innovation_counter = ti.field(dtype=int, shape=())

    def clone_without_network(self):
        result = Population(self.num_inputs, self.num_outputs,
                            self.num_individuals, self.num_repeats)
        result.innovation_counter = self.innovation_counter
        return result

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
            self.links[i].deactivate()
            self.nodes[i].deactivate()
            for _ in range(self.num_inputs):
                self.nodes[i].append(
                    Node(NodeKinds.INPUT.value,
                         activation_funcs.random()))
            for _ in range(self.num_outputs):
                self.nodes[i].append(
                    Node(NodeKinds.OUTPUT.value,
                         activation_funcs.random()))

    # TODO: combine this with the activation functions?
    @ti.func
    def activate_one(self, i, inputs):
        num_nodes = self.nodes[i].length()
        num_links = self.links[i].length()

        # Populate input node activations
        for n in range(self.num_inputs):
            self.nodes[i, n].prev_act = inputs[n]

        # Preserve previous activations before computing the next round.
        # TODO: It'd be faster to swap buffer pointers than to copy.
        for n in range(self.num_inputs, num_nodes):
            self.nodes[i, n].prev_act = self.nodes[i, n].curr_act

        # Compute activations for non-input nodes
        for n in range(self.num_inputs, num_nodes):
            if not self.nodes[i, n].deleted:
                raw = 0.0
                for l in range(num_links):
                    link = self.links[i, l]
                    if not link.deleted and link.to_node == n:
                        value = self.nodes[i, link.from_node].prev_act
                        weight = link.weight
                        raw += value * weight
                next_act = activation_funcs.call(
                    self.nodes[i, n].act_func,
                    raw + self.nodes[i, n].bias)
                self.nodes[i, n].curr_act = next_act

        # Return a vector of the activation values for just the output nodes.
        return ti.Vector([
            self.nodes[i, n].curr_act for n in range(
                self.num_inputs, self.num_inputs + self.num_outputs)])

    @ti.kernel
    def activate_all(self, inputs: ti.template(), outputs: ti.template()):
        for i, r in ti.ndrange(self.num_individuals, self.num_repeats):
            w = i * self.num_repeats + r
            outputs[w] = self.activate_one(i, inputs[w])

    @ti.func
    def render_one_helper(self, w, i, outputs):
        _, rows, cols = outputs.shape
        for row, col in ti.ndrange(rows, cols):
            inputs = ti.Vector([row / rows, col / cols])
            # TODO: Handle outputs of varying sizes, not just 1.
            outputs[w, row, col] = self.activate_one(i, inputs)[0]

    @ti.kernel
    def render_one_kernel(self, w: int, i: int, outputs: ti.template()):
        self.render_one_helper(w, i, outputs)

    def render_one(self, index, shape):
        shape = (1,) + shape
        result = ti.field(float, shape=shape)
        self.render_one_kernel(index, index, result)
        return result.to_numpy()[0]

    @ti.kernel
    def render_all(self, outputs: ti.template()):
        for i, r in ti.ndrange(self.num_individuals, self.num_repeats):
            w = i * self.num_repeats + r
            self.render_one_helper(w, i, outputs)

def random_population(num_inputs, num_outputs, num_individuals, num_repeats):
    result = Population(num_inputs, num_outputs, num_individuals, num_repeats)
    result.randomize_all()
    return result
