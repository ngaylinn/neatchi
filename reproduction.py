import numpy as np
import taichi as ti

from . import activation_funcs
from .data_types import Node, NodeKinds

MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.6

# Constants for testing compatibility between two individuals in a Population.
DISJOINT_COEFF = 1.0
WEIGHT_COEFF = 1.0
COMPATIBILITY_THRESHOLD = 1.0

@ti.func
def add_random_link(pop, i):
    num_nodes = pop.nodes[i].length()
    # Randomly pick any node to be the "from" node for this link.
    from_node = ti.random(dtype=int) % num_nodes
    # Randomly pick a non-input node to be the "to" node for this link.
    to_node = ti.random(dtype=int) % (num_nodes - pop.num_inputs)
    to_node += pop.num_inputs
    pop.new_link(i, from_node, to_node, ti.random())


@ti.func
def mutate_one(pop, i):
    mutation_kind = ti.random(dtype=int) % 7
    # Add node
    if mutation_kind == 0 and pop.has_room_for(i, nodes=1, links=2):
        if pop.links[i].length() == 0:
            add_random_link(pop, i)
        l = ti.random(dtype=int) % pop.links[i].length()
        weight = pop.links[i, l].weight
        from_node = pop.links[i, l].from_node
        to_node = pop.links[i, l].to_node
        pop.links[i, l].deleted = True
        n = pop.nodes[i].length()
        pop.nodes[i].append(Node(NodeKinds.HIDDEN.value,
                                 activation_funcs.random(), ti.random()))
        pop.new_link(i, from_node, n, weight)
        pop.new_link(i, n, to_node, 1.0)

    # Remove node
    elif mutation_kind == 1:
        num_frozen = pop.num_inputs + pop.num_outputs
        num_hidden = pop.nodes[i].length() - num_frozen
        if num_hidden > 0:
            n = ti.random(dtype=int) % num_hidden + num_frozen
            pop.nodes[i, n].deleted = True
            for l in range(pop.links[i].length()):
                link = pop.links[i, l]
                if (link.from_node == n or link.to_node == n):
                    pop.links[i, l].deleted = True

    # Change activation
    if mutation_kind == 2:
        n = ti.random(dtype=int) % pop.nodes[i].length()
        pop.nodes[i, n].act_func = activation_funcs.random()

    # Change bias
    elif mutation_kind == 3:
        n = ti.random(dtype=int) % pop.nodes[i].length()
        pop.nodes[i, n].bias = ti.random()

    ## Add link
    elif mutation_kind == 4 and pop.has_room_for(i, nodes=0, links=1):
        add_random_link(pop, i)

    ## Remove link
    elif mutation_kind == 5 and pop.links[i].length() > 0:
        l = ti.random(dtype=int) % pop.links[i].length()
        pop.links[i, l].deleted = True

    # Change weight
    elif mutation_kind == 6:
        if pop.links[i].length() > 0:
            l = ti.random(dtype=int) % pop.links[i].length()
            pop.links[i, l].weight = ti.random()
        else:
            add_random_link(pop, i)


@ti.data_oriented
class Couplings:
    PARENT = 0
    MATE = 1
    NONE = -1

    def __init__(self, pop, parent_selections, mate_selections):
        self.pop = pop
        self.max_innov = pop.innovation_counter[None]

        # Allocate a workspace for alligning link lists by innovation numbers.
        self.selections = ti.Vector.field(
            n=2, dtype=int, shape=pop.num_individuals)
        self.selections.from_numpy(
            np.stack((parent_selections, mate_selections), axis=1
                    ).astype(np.int32))

        # TODO: Optimize?
        self.innov2l = ti.Vector.field(
            n=2, dtype=int, shape=(pop.num_individuals, self.max_innov + 1))
        self.innov2l.fill(self.NONE)

    @ti.func
    def align_link_lists(self, i):
        p, m = self.couple_indices(i)
        for l in range(self.pop.links[p].length()):
            innov = self.pop.links[p, l].innov
            self.innov2l[innov, self.PARENT] = l
        for l in range(self.pop.links[m].length()):
            innov = self.pop.links[m, l].innov
            self.innov2l[innov, self.MATE] = l

    @ti.func
    def is_compatible(self, i):
        """Returns True iff parent and mate are similar enough to breed."""
        num_disjoint = 0
        num_total = 0
        weight_delta = 0.0

        # Traverse links in order of innovation number
        for v in range(self.max_innov):
            pl, ml = self.link_indices(i, v)
            # If both parent and mate have this link, compare their weights.
            if (pl != self.NONE and ml != self.NONE):
                p, m = self.couple_indices(i)
                num_total += 1
                weight_delta += ti.abs(
                    self.pop.links[p, pl].weight -
                    self.pop.links[m, ml].weight)

            # If just one of parent and mate have this link, it's "disjoint." The
            # original Neat algorithm discriminates between "disjoint" and "excess"
            # genes, but here we treat them all the same.
            elif (pl != self.NONE or ml != self.NONE):
                num_total += 1
                num_disjoint += 1

        return ((DISJOINT_COEFF * num_disjoint / num_total) +
                (WEIGHT_COEFF * weight_delta)) < COMPATIBILITY_THRESHOLD

    @ti.func
    def should_crossover(self, i):
        p, m = self.couple_indices(i)
        result = False
        if p != m and ti.random() < CROSSOVER_RATE:
            self.align_link_lists(i)
            result = self.is_compatible(i)
        return result

    @ti.func
    def link_indices(self, i, v):
        return self.innov2l[i, v].cast(int)

    @ti.func
    def couple_indices(self, i):
        return self.selections[i].cast(int)

    @ti.func
    def crossover(self, output_pop, i):
        # Propagate nodes from the parent to its clone in output_pop.
        for n in range(self.pop.nodes[i].length()):
            output_pop.nodes[i].append(self.pop.nodes[i, n])

        # Traverse links from parent and mate, aligned by innovation number.
        for v in range(self.max_innov):
            pl, ml = self.link_indices(i, v)
            # If parent doesn't have this link, don't include it (in other words,
            # disjoint and excess genes always come from parent, not mate).
            if pl != Couplings.NONE:
                p, m = self.couple_indices(i)
                # If parent and mate both have this link, pick one at random.
                # Otherwise, just use the parent's copy.
                if ml != Couplings.NONE and ti.random(dtype=int) % 2:
                    output_pop.links[i].append(self.pop.links[m, ml])
                else:
                    output_pop.links[i].append(self.pop.links[p, pl])

    @ti.func
    def clone(self, output_pop, i):
        p, _ = self.couple_indices(i)
        # Propagate nodes from this individual to its clone in output_pop.
        for n in range(self.pop.nodes[i].length()):
            output_pop.nodes[i].append(self.pop.nodes[p, n])

        # Propagate links from this individual to its clone in output_pop.
        for l in range(self.pop.links[i].length()):
            if not self.pop.links[i, l].deleted:
                input_link = self.pop.links[p, l]
                output_pop.links[i].append(input_link)

    @ti.func
    def make_offspring(self, output_pop, i):
        # If this couple is compatible and chance is on their side, perform
        # crossover.
        if self.should_crossover(i):
            self.crossover(output_pop, i)
        # Otherwise, just make a clone of the one parent.
        else:
            self.clone(output_pop, i)

        # Finally, mutate the offspring.
        mutate_one(output_pop, i)

    @ti.kernel
    def propagate(self, output_pop: ti.template()):
        for i in range(output_pop.num_individuals):
            self.make_offspring(output_pop, i)


def propagate(input_pop, parent_selections, mate_selections):
    output_pop = input_pop.clone_without_network()
    couplings = Couplings(input_pop, parent_selections, mate_selections)
    couplings.propagate(output_pop)
    return output_pop

