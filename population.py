"""A collection of CPPNs to evolve using the Neat algorithm.

Each population is a collection of CPPNs with the same properties (number of
inputs and outputs, whether the network is recurrent or not). The population
contains large fields of Node and Link objects for all the CPPNs in the
population. These fields are dynamic, which means they grow in size like lists
as the CPPNs evolve. This means that the time spent on reproduction and
activation increases as the CPPNs evolve to be more complex.
"""

import taichi as ti

from .data_types import Link, Node, link_to_str, node_to_str

MAX_NETWORK_SIZE = 2**8


@ti.data_oriented
class NeatPopulation:
    def __init__(self, num_inputs, num_outputs, num_individuals, is_recurrent,
                 innovation_counter):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_individuals = num_individuals
        self.is_recurrent = is_recurrent
        self.innovation_counter = innovation_counter

        node_lists = ti.root.dense(ti.i, self.num_individuals) \
                .dynamic(ti.j, MAX_NETWORK_SIZE, chunk_size=32)
        self.nodes = Node.field()
        node_lists.place(self.nodes)

        link_lists = ti.root.dense(ti.i, self.num_individuals) \
                .dynamic(ti.j, MAX_NETWORK_SIZE, chunk_size=32)
        self.links = Link.field()
        link_lists.place(self.links)

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
        """Adds a link with a new innovation number, for mutations."""
        innov = ti.atomic_add(self.innovation_counter[None], 1)
        link = Link(from_node, to_node, weight, False, innov)
        self.links[i].append(link)

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
