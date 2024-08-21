"""A collection of CPPNs to evolve using the Neat algorithm.

Each population is a collection of CPPNs with the same properties (number of
inputs and outputs, whether the network is recurrent or not). The population
contains large fields of Node and Link objects for all the CPPNs in the
population. These fields are dynamic, which means they grow in size like lists
as the CPPNs evolve. This means that the time spent on reproduction and
activation increases as the CPPNs evolve to be more complex.
"""

from abc import ABC, abstractmethod

import numpy as np
import taichi as ti

from .activation import activate_network
from .data_types import export_cppns, import_cppns, Link, MAX_NETWORK_SIZE, Node
from .reproduction import mutate_all, propagate, random_init
from .selection import get_compatibility
from .validation import validate_all


@ti.data_oriented
class Population(ABC):
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def __init__(self, population_shape, index, matchmaker=None):
        self.population_shape = population_shape
        self.num_sub_pops, self.num_individuals = population_shape

        if isinstance(index, np.ndarray):
            self.index = ti.Vector.field(n=2, dtype=int, shape=len(index))
            self.index.from_numpy(index)
        elif isinstance(index, ti.lang.matrix.MatrixField):
            self.index = index

        self.matchmaker = matchmaker

    @abstractmethod
    @ti.func
    def activate(self, w, inputs):
        ...

    def randomize(self, elites=None):
        self.matchmaker.reset()
        self.make_random_population(elites)
        self.matchmaker.analyze_compatibility(self, 0)

    # NOTE: Value of generation must be less than history_size
    def propagate(self, generation=0):
        self.matchmaker.update_matches(generation)
        self.make_next_generation(generation)
        self.matchmaker.analyze_compatibility(self, generation + 1)

    @abstractmethod
    def to_numpy(self):
        ...

    @abstractmethod
    def from_numpy(self, arr):
        ...

    # -------------------------------------------------------------------------
    # Internal API
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_compatibility(self, sp, p, m):
        ...

    @abstractmethod
    def make_next_generation(self, generation):
        ...

    @abstractmethod
    def make_random_population(self, elites):
        ...


# files.
@ti.data_oriented
class CppnPopulation(Population):
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def __init__(self, population_shape, network_shape, index,
                 matchmaker=None):
        super().__init__(population_shape, index, matchmaker)

        self.num_inputs, self.num_outputs = network_shape

        # Used to track when links were added to aid in crossover.
        self.innovation_counter = ti.field(dtype=int, shape=())

        # Used to manage a double buffer of CPPNs: one for the current
        # generation, and one for the generation produced during propagation.
        self.buffer_index = ti.field(dtype=int, shape=())
        NUM_BUFFERS = 2

        # Allocate fields to track the node and link lists for every CPPN in
        # the population.
        # NOTE: It's a shame, but Taichi's dynamic fields are just too awkward
        # to use the way I want here. So, I'm implementing list-like behavior
        # manually.
        # NOTE: This is inefficient and ill-suited to the GPU, but still much
        # better than doing this work on the CPU.
        self.nodes = Node.field(
            shape=(NUM_BUFFERS, self.num_sub_pops, self.num_individuals,
                   MAX_NETWORK_SIZE))
        self.node_lens = ti.field(int,
            shape=(NUM_BUFFERS, self.num_sub_pops, self.num_individuals))

        self.links = Link.field(
            shape=(NUM_BUFFERS, self.num_sub_pops, self.num_individuals,
                   MAX_NETWORK_SIZE))
        self.link_lens = ti.field(int,
            shape=(NUM_BUFFERS, self.num_sub_pops, self.num_individuals))

    def to_numpy(self):
        return export_cppns(self)

    def from_numpy(self, cppns):
        import_cppns(self, cppns)

    @ti.func
    def activate(self, w, inputs):
        return activate_network(self, w, inputs)

    # -------------------------------------------------------------------------
    # Internal API: Node management
    # -------------------------------------------------------------------------

    @ti.func
    def num_nodes(self, sp, i):
        b = self.buffer_index[None]
        return self.node_lens[b, sp, i]

    @ti.func
    def get_node(self, sp, i, n):
        b = self.buffer_index[None]
        return self.nodes[b, sp, i, int(n)]

    @ti.func
    def set_node(self, sp, i, n, node):
        b = self.buffer_index[None]
        self.nodes[b, sp, i, int(n)] = node

    @ti.func
    def insert_node(self, sp, i, n, node):
        # NOTE: This function is O(N+L)
        # It may be worthwhile to optimize this in the future, but for now it
        # seems better to make this as simple as possible to minize bugs.
        b = self.buffer_index[None]
        num_nodes = self.num_nodes(sp, i)
        # We will shift down a contiguous block of nodes to make room for the new
        # one, so make variables to mark the begining and end of that block.
        shift_begin, shift_end = n, n
        # Walk along the node list, starting from the insertion point all the way
        # to the one beyond the end of the list (one beyond because we are
        # extending this list by one).
        for n2 in range(n, num_nodes + 1):
            shift_end = n2
            # If we've reached the end of the list, just append the last node.
            if n2 == num_nodes:
                self.nodes[b, sp, i, n2] = node
                self.node_lens[b, sp, i] += 1
            # Otherwise, insert the node in this position, take whatever node used
            # to be in that spot, and continue shifting it down.
            else:
                self.nodes[b, sp, i, n2], node = node, self.nodes[b, sp, i, n2]

        # Now that the nodes are in order, go back through the links and update any
        # references to nodes that got shifted.
        for l in range(self.num_links(sp, i)):
            link = self.links[b, sp, i, l]
            if link.from_node >= shift_begin and link.from_node < shift_end:
                self.links[b, sp, i, l].from_node += ti.cast(1, ti.int8)
            if link.to_node >= shift_begin and link.to_node < shift_end:
                self.links[b, sp, i, l].to_node += ti.cast(1, ti.int8)

    @ti.func
    def delete_node(self, sp, i, n):
        # NOTE: This function is O(N+L^2)
        # It may be worthwhile to optimize this in the future, but for now it
        # seems better to make this as simple as possible to minize bugs.
        b = self.buffer_index[None]
        num_nodes = self.num_nodes(sp, i)
        # Walk along the node lists, shifting each node forward into the
        # position that was just vacated.
        for n2 in range(n, num_nodes - 1):
            self.nodes[b, sp, i, n2] = self.nodes[b, sp, i, n2 + 1]
        self.node_lens[b, sp, i] -= 1

        # Now update any links that have been affected. This is a while loop
        # instead of a for loop because we may delete links as we iterate,
        # causing the loop bounds and index to change.
        l = 0
        while l < self.num_links(sp, i):
            link = self.links[b, sp, i, l]
            # Any links referring to nodes after this one need their indices
            # updated to reflect the deletion.
            if link.from_node > n:
                self.links[b, sp, i, l].from_node -= ti.cast(1, ti.int8)
            if link.to_node > n:
                self.links[b, sp, i, l].to_node -= ti.cast(1, ti.int8)
            # Any links to or from the deleted node are deleted.
            if link.from_node == n or link.to_node == n:
                self.delete_link(sp, i, l)
                # Prevent the link index from incrementing so that we don't
                # skip the link after the one we deleted.
                continue
            l += 1

    # -------------------------------------------------------------------------
    # Internal API: Link management
    # -------------------------------------------------------------------------

    @ti.func
    def num_links(self, sp, i):
        b = self.buffer_index[None]
        return self.link_lens[b, sp, i]

    @ti.func
    def get_link(self, sp, i, l):
        b = self.buffer_index[None]
        return self.links[b, sp, i, l]

    @ti.func
    def set_link(self, sp, i, l, link):
        b = self.buffer_index[None]
        self.links[b, sp, i, l] = link

    @ti.func
    def add_link(self, sp, i, link):
        b = self.buffer_index[None]
        link.innov = ti.atomic_add(self.innovation_counter[None], 1)
        self.links[b, sp, i, self.num_links(sp, i)] = link
        self.link_lens[b, sp, i] += 1

    @ti.func
    def delete_link(self, sp, i, l):
        # NOTE: This function is O(L)
        # It may be worthwhile to optimize this in the future, but for now it
        # seems better to make this as simple as possible to minize bugs.
        b = self.buffer_index[None]
        num_links = self.num_links(sp, i)
        for l2 in range(l, num_links - 1):
            self.links[b, sp, i, l2] = self.links[b, sp, i, l2 + 1]
        self.link_lens[b, sp, i] -= 1

    # -------------------------------------------------------------------------
    # Internal API: Other
    # -------------------------------------------------------------------------

    @ti.func
    def get_compatibility(self, sp, p, m):
        return get_compatibility(self, sp, p, m)

    @ti.kernel
    def clear(self):
        # Reset all the node and link lists to empty for the current population
        # buffer only.
        b = self.buffer_index[None]
        for sp, i in ti.ndrange(*self.population_shape):
            self.node_lens[b, sp, i] = 0
            self.link_lens[b, sp, i] = 0

    @ti.kernel
    def rotate_buffer(self):
        # For some reason, Taichi gives a type cast warning if this is done
        # from Python scope.
        self.buffer_index[None] = 1 - self.buffer_index[None]

    @ti.func
    def has_room_for(self, sp, i, nodes, links):
        return (self.num_nodes(sp, i) + nodes < MAX_NETWORK_SIZE and
                self.num_nodes(sp, i) + links < MAX_NETWORK_SIZE)

    @ti.kernel
    def restore_elites(self, elites: ti.types.ndarray(dtype=ti.int32, ndim=2)):
        b_curr = self.buffer_index[None]
        b_prev = 1 - b_curr

        # Copy the full network from the previous buffer to the current one,
        # overriding whatever was generated from the usual breeding program.
        for e, x in ti.ndrange(elites.shape[0], MAX_NETWORK_SIZE):
            sp, i = elites[e, 0], elites[e, 1]
            self.nodes[b_curr, sp, i, x] = self.nodes[b_prev, sp, i, x]
            self.links[b_curr, sp, i, x] = self.links[b_prev, sp, i, x]

            # Also update the match-making records and list lengths to reflect
            # this change. This only needs to happen once per individual.
            if x == 0:
                self.node_lens[b_curr, sp, i] = self.node_lens[b_prev, sp, i]
                self.link_lens[b_curr, sp, i] = self.link_lens[b_prev, sp, i]

    def make_random_population(self, elites=None):
        # Rotate the population buffer so that the previous population won't be
        # clobbered and we can copy elites from there.
        self.rotate_buffer()

        # Initialize all the CPPNs in the current generation.
        self.clear()
        random_init(self)

        # Copy back the elites from the previous population.
        if elites is not None:
            self.restore_elites(elites)

        # For debugging, make sure all the new CPPNs are valid.
        # validate_all(self)

    def make_next_generation(self, generation=0):
        # Fill the unused population buffer with a new generation drawn from
        # the current population buffer, possibly with crossover.
        propagate(self, generation)

        # Swap the buffers to make the new generation current, then mutate that
        # generation. This is done after the propagate step so that the code
        # for mutation doesn't need to be aware of what buffer index to use.
        self.rotate_buffer()
        mutate_all(self)

        # For debugging, make sure all the new CPPNs are valid.
        # validate_all(self)
