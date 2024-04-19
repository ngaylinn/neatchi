import numpy as np
import taichi as ti

from . import activation_funcs
from .data_types import Link, Node, NodeKinds, link_to_str, node_to_str

# Max sizes used to construct efficient lookup tables in reproduction.py
# TODO: I tore out this optimization because it was causing annoying implicit
# cast warnings. Add it back if it seems worthwhile.
# NETWORK_INDEX_DTYPE = ti.uint8
MAX_NETWORK_SIZE = 2**8
MAX_INNOVATIONS = 2**16


@ti.data_oriented
class NeatPopulation:
    def __init__(self, num_inputs, num_outputs, num_individuals):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_individuals = num_individuals

        node_lists = ti.root.dense(ti.i, self.num_individuals) \
                .dynamic(ti.j, MAX_NETWORK_SIZE, chunk_size=32)
        self.nodes = Node.field()
        node_lists.place(self.nodes)

        link_lists = ti.root.dense(ti.i, self.num_individuals) \
                .dynamic(ti.j, MAX_NETWORK_SIZE, chunk_size=32)
        self.links = Link.field()
        link_lists.place(self.links)

        # TODO: Share one across all populations?
        self.innovation_counter = ti.field(dtype=int, shape=())

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

    @ti.kernel
    def size_of(self, i: int) -> ti.types.vector(2, int):
        return ti.Vector([self.nodes[i].length(), self.links[i].length()])

    def print_one(self, i):
        def print_edge(num_cells, top):
            print('┏' if top else '┗', end='')
            for cell in range(num_cells):
                if cell > 0:
                    print('┳' if top else '┻', end='')
                print('━' * 8, end='')
            print('┓' if top else '┛')

        def print_body(row_strs):
            for row in range(len(row_strs[0])):
                for cell in range(len(row_strs)):
                    print('┃' + row_strs[cell][row], end='')
                print('┃')

        num_nodes, num_links = self.size_of(i)
        print_edge(num_nodes, top=True)
        print_body([node_to_str(self.nodes[i, n]).split('\n')
                    for n in range(num_nodes)])
        print_edge(num_nodes, top=False)

        if num_links == 0:
            return
        print_edge(num_links, top=True)
        print_body([link_to_str(self.links[i, l]).split('\n')
                    for l in range(num_links)])
        print_edge(num_links, top=False)

    def print_all(self):
        for i in range(self.num_individuals):
            print(f'Individual {i}')
            self.print_one(i)
            print()

def random_population(num_inputs, num_outputs, num_individuals):
    result = NeatPopulation(num_inputs, num_outputs, num_individuals)
    result.randomize_all()
    return result
