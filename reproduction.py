import taichi as ti

from . import activation_funcs
from . import population
from .data_types import Node, NodeKinds

MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.6

# Constants for testing compatibility between two individuals in a Population.
# TODO: Tune these!
DISJOINT_COEFF = 1.0
WEIGHT_COEFF = 1.0
COMPATIBILITY_THRESHOLD = 1.0

@ti.data_oriented
class Matches:
    """Data structure for tracking parent / mate pairs and their link lists."""

    NONE = -1

    def __init__(self, num_individuals):
        print(f'A Matches {id(self)}')
        self.selections = ti.Vector.field(
            n=2, dtype=int, shape=num_individuals)
        self.mate_links = ti.field(
            int, shape=(num_individuals, population.MAX_NETWORK_SIZE))

    def __del__(self):
        print(f'D Matches {id(self)}')

    @ti.kernel
    def update_mate_links(self, pop: ti.template()):
        for i in range(pop.num_individuals):
            p, m = self.selections[i]
            for pl in range(pop.links[p].length()):
                p_innov = pop.links[p, pl].innov
                for ml in range(pop.links[m].length()):
                    m_innov = pop.links[m, ml].innov
                    if (p_innov == m_innov):
                        self.mate_links[p, pl] = ml

    def update(self, pop, selections):
        self.selections.from_numpy(selections)
        self.mate_links.fill(self.NONE)
        self.update_mate_links(pop)

    @ti.func
    def is_compatible(self, input_pop, i):
        """Returns True iff parent and mate are similar enough to breed."""
        num_disjoint = 0
        num_total = 0
        weight_delta = 0.0

        p, m = self.selections[i]
        for pl in range(input_pop.links[p].length()):
            ml = self.mate_links[p, pl]
            # If both parent and mate have this link, compare their weights.
            if (pl != self.NONE and ml != self.NONE):
                num_total += 1
                weight_delta += ti.abs(
                    input_pop.links[p, pl].weight -
                    input_pop.links[m, ml].weight)

            # If just one of parent and mate have this link, it's "disjoint." The
            # original Neat algorithm discriminates between "disjoint" and "excess"
            # genes, but here we treat them all the same.
            elif (pl != self.NONE or ml != self.NONE):
                num_total += 1
                num_disjoint += 1

        return ((DISJOINT_COEFF * num_disjoint / num_total) +
                (WEIGHT_COEFF * weight_delta)) < COMPATIBILITY_THRESHOLD

    @ti.func
    def should_crossover(self, input_pop, i):
        p, m = self.selections[i]
        result = False
        if p != m and ti.random() < CROSSOVER_RATE:
            result = self.is_compatible(input_pop, i)
        return result


@ti.func
def add_random_link(pop, i):
    num_nodes = pop.nodes[i].length()
    # Randomly pick any node to be the "from" node for this link.
    from_node = ti.random(dtype=int) % num_nodes
    # Randomly pick a non-input node to be the "to" node for this link.
    to_node = ti.random(dtype=int) % (num_nodes - pop.num_inputs)
    to_node += pop.num_inputs
    pop.new_link(i, from_node, to_node, ti.random())


# TODO: Optimize? This is inherently inefficient, since different threads are
# doing completely different work, but it may be possible to speed up by
# reducing the length of the longest branch or by reorienting the computation
# so we apply the same mutation to many genes at once.
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

@ti.func
def crossover(input_pop, output_pop, i, matches):
    p, m = matches.selections[i]

    # Propagate nodes from the parent to its clone in output_pop.
    for n in range(input_pop.nodes[p].length()):
        output_pop.nodes[i].append(input_pop.nodes[p, n])

    for pl in range(input_pop.links[p].length()):
        ml = matches.mate_links[p, pl]
        # If parent doesn't have this link, don't include it (in other words,
        # disjoint and excess genes always come from parent, not mate).
        if pl != Matches.NONE:
            # If parent and mate both have this link, pick one at random.
            # Otherwise, just use the parent's copy.
            if ml != Matches.NONE and ti.random(dtype=int) % 2:
                output_pop.links[i].append(input_pop.links[m, ml])
            else:
                output_pop.links[i].append(input_pop.links[p, pl])

@ti.func
def clone(input_pop, output_pop, i, matches):
    p, _ = matches.selections[i]

    # Propagate nodes from this individual to its clone in output_pop.
    for n in range(input_pop.nodes[p].length()):
        output_pop.nodes[i].append(input_pop.nodes[p, n])

    # Propagate links from this individual to its clone in output_pop.
    for l in range(input_pop.links[p].length()):
        input_link = input_pop.links[p, l]
        if not input_link.deleted:
            output_pop.links[i].append(input_link)

@ti.kernel
def propagate(input_pop:ti.template(), output_pop: ti.template(),
              matches: ti.template()):
    for i in range(output_pop.num_individuals):
        # If this couple is compatible and chance is on their side, perform
        # crossover.
        if matches.should_crossover(input_pop, i):
            crossover(input_pop, output_pop, i, matches)
        # Otherwise, just make a clone of the one parent.
        else:
            clone(input_pop, output_pop, i, matches)

        # Finally, mutate the offspring.
        mutate_one(output_pop, i)
