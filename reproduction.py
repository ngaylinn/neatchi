"""Code for random initialization, crossover, mutations, etc.

This module is used to populate and modify the CPPNs in NeatPopulation
objects. The random_init() function generates one from scratch, while the
propagate() function generates one from the previous population, either
cloning or performing crossover on selected individuals.

All new individuals in the population are subject to mutations, which modify
the structure of their CPPN by adding, removing, or changing neurons (nodes) or
synapses (links). These operations should always produce valid CPPNs, with
nodes sorted in activation order for non-recurrent networks. The validate*()
functions can be used to confirm the new CPPNs are all consistent and correct,
and serves to document the invariants for a well-formed CPPN.

This implementation of the NEAT algorithm enforces that only "compatible"
individuals are bred together, as determined by Matches.is_compatible(), which
computes the weighted sum of a few difference metrics. It does not explicitly
break the population into species, since this is not desirable in all cases.
Since selection is handled by the caller, it is possible to implement this
behavior manually. In the future, a standard implementation may be provided.
"""

import taichi as ti

from . import activation_funcs
from . import population
from .data_types import Node, NodeKinds

MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.6

# TODO: Are these ranges reasonable?
BIAS_RANGE = 1.0
GAIN_RANGE = 8.0
WEIGHT_RANGE = 1.0

# Constants for testing compatibility between two individuals in a Population.
# TODO: Tune these!
# TODO: Maybe compute compatibility pairwise for many individuals, so that
# selection can pick a compatible mate rather than merely filtering out
# incompatible selections after the fact.
# TODO: Capture and report compatibility / diversity metrics for the
# population?
DISJOINT_COEFF = 1.0
WEIGHT_COEFF = 1.0
COMPATIBILITY_THRESHOLD = 1.0

@ti.func
def rand_range(min_val, max_val):
    """Pick a random int from the given range."""
    return ti.random(dtype=int) % (max_val - min_val) + min_val


# NOTE: Taichi's assert statement provides minimal useful information and isn't
# evaluated until after the kernel completes, so I need all these verbose if
# statements instead. Gross.
@ti.func
def validate(pop, i):
    """Check all invariants on the given NN, for debugging code in this file.

    Recommend running this with ti.init(arch=ti.cpu, cpu_max_num_threads=1) to
    ensure the last message on screen corresponds to the error that caused the
    assert to fail. Uncommon print statements below to see which node / link in
    the population triggered the error and why.
    """
    num_nodes = pop.nodes[i].length()
    num_links = pop.links[i].length()
    valid = True
    for n in range(num_nodes):
        node = pop.nodes[i, n]
        print('validating node', i, n, node.kind, node.act_func, node.bias, node.gain, node.deleted)
        if n < pop.num_inputs:
            if node.kind != NodeKinds.INPUT.value:
                print('Validation error: node', n, ', node.kind != NodeKinds.INPUT.value')
                valid = False
            if node.deleted:
                print('Validation error: node', n, ', node.deleted')
                valid = False
        elif n < pop.nodes[i].length() - pop.num_outputs:
            if node.kind != NodeKinds.HIDDEN.value:
                print('Validation error: node', n, ', node.kind != NodeKinds.HIDDEN.value')
                valid = False
        else:
            if node.kind != NodeKinds.OUTPUT.value:
                print('Validation error: node', n, ', node.kind != NodeKinds.OUTPUT.value')
                valid = False
            if node.deleted:
                print('Validation error: node', n, ', node.deleted')
                valid = False
            if node.act_func < 0:
                print('Validation error: node', n, ', node.act_func < 0')
                valid = False
            if node.act_func >= activation_funcs.NUM_ACTIVATION_FUNCS:
                print('Validation error: node', n, ', node.act_func >= activation_funcs.NUM_ACTIVATION_FUNCS')
                valid = False
            if node.bias < -BIAS_RANGE:
                print('Validation error: node', n, ', node.bias <', -BIAS_RANGE)
                valid = False
            if node.bias > BIAS_RANGE:
                print('Validation error: node', n, ', node.bias >', BIAS_RANGE)
                valid = False
            if node.gain < -GAIN_RANGE:
                print('Validation error: node', n, ', node.gain <', -GAIN_RANGE)
                valid = False
            if node.gain > GAIN_RANGE:
                print('Validation error: node', n, ', node.gain >', GAIN_RANGE)
                valid = False
    for l in range(num_links):
        link = pop.links[i, l]
        print('validating link', i, l, link.from_node, link.to_node, link.weight, link.deleted, link.innov)
        if link.from_node < 0:
            print('Validation error: link', l, ', link.from_node < 0')
            valid = False
        if link.from_node >= num_nodes:
            print('Validation error: link', l, ', link.from_node >= num_nodes')
            valid = False
        if link.to_node < pop.num_inputs:
            print('Validation error: link', l, ', link.to_node < pop.num_inputs')
            valid = False
        if link.to_node >= num_nodes:
            print('Validation error: link', l, ', link.to_node >= num_nodes')
            valid = False
        if not pop.is_recurrent:
            if link.to_node <= link.from_node:
                print('Validation error: link', l, ', link.to_node <= link.from_node')
                valid = False
            if pop.nodes[i, link.from_node].deleted:
                print('Validation error: link', l, ', pop.nodes[i, link.from_node].deleted')
                valid = False
            if pop.nodes[i, link.to_node].deleted:
                print('Validation error: link', l, ', pop.nodes[i, link.to_node].deleted')
                valid = False
            if link.weight < -WEIGHT_RANGE:
                print('Validation error: link', l, ', link.weight <', -WEIGHT_RANGE)
                valid = False
            if link.weight > WEIGHT_RANGE:
                print('Validation error: link', l, ', link.weight >', WEIGHT_RANGE)
                valid = False
            if link.innov >= pop.innovation_counter[None]:
                print('Validation error: link', l, ', link.innov >= pop.innovation_counter[None]')
                valid = False
        for l2 in range(num_links):
            if l != l2:
                if link.innov == pop.links[i, l2].innov:
                    print('Validation error: link', l, ', link.innov == pop.links[i, l2].innov')
                    valid = False
    assert valid

@ti.kernel
def validate_all_kernel(pop: ti.template()):
    for i in range(pop.num_individuals):
        validate(pop, i)

def validate_all(pop):
    validate_all_kernel(pop)


@ti.data_oriented
class Matches:
    """Data structure for tracking parent / mate pairs and their link lists."""

    NONE = -1

    def __init__(self, num_individuals):
        self.selections = ti.Vector.field(
            n=2, dtype=int, shape=num_individuals)
        # A table mapping the indices for each link in each parent to the
        # corresponding link index in that parent's mate (if there is one).
        self.mate_links = ti.field(
            int, shape=(num_individuals, population.MAX_NETWORK_SIZE))

    @ti.kernel
    def update_mate_links(self, pop: ti.template()):
        # For each individual child
        for i in range(pop.num_individuals):
            # Find its selected parent / parent's mate
            p, m = self.selections[i]
            # For all of parent's links
            for pl in range(pop.links[p].length()):
                p_link = pop.links[p, pl]
                if not p_link.deleted:
                    continue
                # Compare with all of mate's links
                for ml in range(pop.links[m].length()):
                    m_link = pop.links[m, ml]
                    if m_link.deleted:
                        continue
                    # Links correspond if they have the same innovation number.
                    if (p_link.innov == m_link.innov):
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
            # original NEAT algorithm discriminates between "disjoint" and "excess"
            # genes, but here we treat them all the same.
            elif (pl != self.NONE or ml != self.NONE):
                num_total += 1
                num_disjoint += 1

        return ((DISJOINT_COEFF * num_disjoint / num_total) +
                (WEIGHT_COEFF * weight_delta)) < COMPATIBILITY_THRESHOLD

    @ti.func
    def should_crossover(self, input_pop, i):
        p, m = self.selections[i]
        # Reproduction is asexual by default, but if selection chose a distinct
        # parent and mate, they are compatible with each other, and luck is on
        # their side, then perform sexual reproduction with crossover.
        result = False
        if p != m and ti.random() < CROSSOVER_RATE:
            result = self.is_compatible(input_pop, i)
        return result


@ti.func
def add_random_link(pop, i):
    """Add a random link to the specified CPPN as a new innovation."""
    num_nodes = pop.nodes[i].length()
    num_inputs = pop.num_inputs

    from_node, to_node = 0, 0
    # If this network is recurrent, then randomly pick any two nodes and make a
    # link between them, but disallow links to input nodes.
    if pop.is_recurrent:
        from_node = rand_range(0, num_nodes)
        to_node = rand_range(num_inputs, num_nodes)
    # Otherwise, make sure from_node < to_node and disallow links to input
    # nodes.
    else:
        from_node = rand_range(0, num_nodes - 1)
        to_node = rand_range(ti.max(from_node + 1, num_inputs), num_nodes)

    # Actually make the link, and make sure it gets an innovation number.
    pop.new_link(i, from_node, to_node, ti.random())


@ti.func
def make_random_node(node_type):
    return Node(
        node_type,
        activation_funcs.random(),
        2 * BIAS_RANGE * ti.random() - BIAS_RANGE,
        2 * GAIN_RANGE * ti.random() - GAIN_RANGE,
        False)


@ti.func
def insert_node(pop, i, n, node):
    """Add a node to a CPPN, keeping the node list sorted (slow!)."""
    num_nodes = pop.nodes[i].length()
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
            pop.nodes[i].append(node)
        # If the next spot in the list is free, claim it and stop shifting.
        elif pop.nodes[i, n2].deleted:
            pop.nodes[i, n2] = node
            break
        # Otherwise, insert the node in this position, take whatever node used
        # to be in that spot, and continue shifting it down.
        else:
            pop.nodes[i, n2], node = node, pop.nodes[i, n2]

    # Now that the nodes are in order, go back through the links and update any
    # references to nodes that got shifted.
    for l in range(pop.links[i].length()):
        link = pop.links[i, l]
        if link.from_node >= shift_begin and link.from_node <= shift_end:
            pop.links[i, l].from_node += 1
        if link.to_node >= shift_begin and link.to_node <= shift_end:
            pop.links[i, l].to_node += 1

@ti.func
def add_random_node(pop, i):
    """Replace one link with a node and two links in the same place."""
    # We add nodes by splitting a link, so make sure one is available.
    if pop.links[i].length() == 0:
        add_random_link(pop, i)

    # Select a random link to be split in two with the new node between the two
    # halves.
    l = rand_range(0, pop.links[i].length())
    old_link = pop.links[i, l]

    # Don't split a deleted link.
    # TODO: This means we will actually add new nodes less often as the
    # networks grow in size, which is undesirable!
    if not old_link.deleted:
        # Make a node to go between the nodes that link connected, and allocate
        # a variable for its index (to be determined below).
        node = make_random_node(NodeKinds.HIDDEN.value)
        n = 0

        # For a recurrent network, node order doesn't matter, so just append it
        # at the end.
        if pop.is_recurrent:
            n = pop.nodes[i].length()
            pop.nodes[i].append(node)
        # Otherwise, we must keep nodes in activation order. Pick a spot to
        # insert this new node, shift the others out of the way, and insert.
        # NOTE: This is inefficient, but it makes activation much simpler and
        # more efficient. If MAX_NETWORK_SIZE increases significantly, it may
        # be worth revisiting this design.
        else:
            # Pick a place to put this node, someplace after the from_node and
            # before the to_node (though we might take its place). To respect
            # the node order invariant, the new index must be between the last
            # input node and the first output node.
            n = rand_range(
                ti.max(old_link.from_node, pop.num_inputs - 1),
                ti.min(old_link.to_node, pop.nodes[i].length() - pop.num_outputs)
            ) + 1
            insert_node(pop, i, n, node)

        # Mark the node we chose as deleted.
        pop.links[i, l].deleted = True
        # Insert may have updated node indexes, so refresh our local copy of
        # this link to make sure from_node and to_node are up to date.
        old_link = pop.links[i, l]
        # Add new links (with new innovation numbers) linking the old from_node
        # to the new node (with index n) to the old to_node.
        pop.new_link(i, old_link.from_node, n, old_link.weight)
        pop.new_link(i, n, old_link.to_node, 1.0)

# TODO: Optimize? This is inherently inefficient, since different threads are
# doing completely different work, but it may be possible to speed up by
# reducing the length of the longest branch or by reorienting the computation
# so we apply the same mutation to many genes at once.
# TODO: Several of these mutation operations do not account for deleted links
# and nodes, which means the true rate of mutations will actually decrease as
# the networks grow in size. Fix this! Unfortunately, the best way might be to
# rebuild the lists in every generation, eliminating deleted entries.
@ti.func
def mutate_one(pop, i):
    """Randomly choose one kind of mutation and apply it."""
    mutation_kind = rand_range(0, 8)

    # Add node
    # NOTE: no mutation is applied if the CPPN is already at max size.
    if mutation_kind == 0 and pop.has_room_for(i, nodes=1, links=2):
        add_random_node(pop, i)

    # Remove node
    # NOTE: no mutation is applied if the CPPN has no hidden nodes.
    elif mutation_kind == 1:
        num_hidden = pop.nodes[i].length() - pop.num_inputs - pop.num_outputs
        if num_hidden > 0:
            n = rand_range(pop.num_inputs, pop.num_inputs + num_hidden)
            pop.nodes[i, n].deleted = True
            # Also delete any links that refer to this node.
            for l in range(pop.links[i].length()):
                link = pop.links[i, l]
                if (link.from_node == n or link.to_node == n):
                    pop.links[i, l].deleted = True

    # Change activation
    elif mutation_kind == 2:
        n = rand_range(0, pop.nodes[i].length())
        pop.nodes[i, n].act_func = activation_funcs.random()

    # Change bias
    elif mutation_kind == 3:
        n = rand_range(0, pop.nodes[i].length())
        pop.nodes[i, n].bias = ti.math.clamp(
            pop.nodes[i, n].bias + ti.randn() * BIAS_RANGE,
            -BIAS_RANGE, BIAS_RANGE)

    # Change gain
    elif mutation_kind == 3:
        n = rand_range(0, pop.nodes[i].length())
        pop.nodes[i, n].gain = ti.math.clamp(
            pop.nodes[i, n].gain + ti.randn() *  GAIN_RANGE,
            -GAIN_RANGE, GAIN_RANGE)

    # Add link
    # NOTE: no mutation is applied if the CPPN is already at max size.
    elif mutation_kind == 5 and pop.has_room_for(i, nodes=0, links=1):
        add_random_link(pop, i)

    # Remove link
    # NOTE: no mutation is applied if the CPPN has no links.
    elif mutation_kind == 6 and pop.links[i].length() > 0:
        l = rand_range(0, pop.links[i].length())
        pop.links[i, l].deleted = True

    # Change weight
    elif mutation_kind == 7:
        if pop.links[i].length() > 0:
            l = rand_range(0, pop.links[i].length())
            pop.links[i, l].weight = ti.math.clamp(
                pop.links[i, l].weight + ti.randn() * WEIGHT_RANGE,
                -WEIGHT_RANGE, WEIGHT_RANGE)
        # If there are no links, add a random link instead instead of not
        # applying any mutation.
        else:
            add_random_link(pop, i)

@ti.func
def crossover(input_pop, output_pop, i, matches):
    p, m = matches.selections[i]

    # Propagate nodes from the parent to its clone in output_pop.
    for n in range(input_pop.nodes[p].length()):
        output_pop.nodes[i].append(input_pop.nodes[p, n])

    for pl in range(input_pop.links[p].length()):
        # Look up this link in parent and skip if it's been deleted.
        p_link = input_pop.links[p, pl]
        if p_link.deleted:
            continue
        link = p_link

        # Look to see if there is a corresponding link in mate.
        ml = matches.mate_links[p, pl]
        if ml != Matches.NONE:
            m_link = input_pop.links[m, ml]
            # If that corresponding link isn't deleted, then maybe use it
            # instead of parent's copy of this link (50% chance) .
            if not m_link.deleted and ti.random(dtype=int) % 2:
                link = m_link

        # Append a copy of this link from either parent or mate.
        output_pop.links[i].append(link)

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
def propagate(input_pop: ti.template(), output_pop: ti.template(),
              matches: ti.template()):
    for i in range(output_pop.num_individuals):
        # If this couple is compatible and chance is on their side, perform
        # crossover.
        if matches.should_crossover(input_pop, i):
            crossover(input_pop, output_pop, i, matches)
        # Otherwise, just make a clone of the one parent.
        else:
            clone(input_pop, output_pop, i, matches)

        # Finally, maybe mutate the offspring.
        if ti.random() < MUTATION_RATE:
            mutate_one(output_pop, i)

@ti.kernel
def random_init(pop: ti.template()):
    # Generate a population of individuals with no hidden nodes, random
    # activation functions, and one random mutation each.
    for i in range(pop.num_individuals):
        pop.links[ti.cast(i, int)].deactivate()
        pop.nodes[ti.cast(i, int)].deactivate()
        for _ in range(pop.num_inputs):
            pop.nodes[i].append(make_random_node(NodeKinds.INPUT.value))
        for _ in range(pop.num_outputs):
            pop.nodes[i].append(make_random_node(NodeKinds.OUTPUT.value))
        # Add a randomized number of mutations to make the initial population
        # somewhat diverse.
        for _ in range(rand_range(1, 9)):
            mutate_one(pop, i)
