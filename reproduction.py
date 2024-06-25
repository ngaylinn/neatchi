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
from .data_types import Node, NodeKinds, Link

MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.6

NONE = -1

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


@ti.func
def add_random_link(pop, sp, i):
    """Add a random link to the specified CPPN as a new innovation."""
    num_nodes = pop.num_nodes(sp, i)
    num_inputs = ti.static(pop.network_shape[0])

    from_node, to_node = 0, 0
    # Make sure from_node < to_node and disallow links to input nodes.
    from_node = rand_range(0, num_nodes - 1)
    to_node = rand_range(ti.max(from_node + 1, num_inputs), num_nodes)

    # Actually make the link, and make sure it gets an innovation number.
    pop.add_link(sp, i, Link(from_node, to_node, ti.random()))


@ti.func
def make_random_node(node_type):
    return Node(
        node_type,
        activation_funcs.random(),
        2 * BIAS_RANGE * ti.random() - BIAS_RANGE,
        2 * GAIN_RANGE * ti.random() - GAIN_RANGE,
        False)


@ti.func
def add_random_node(pop, sp, i):
    """Replace one link with a node and two links in the same place."""
    # We add nodes by splitting a link, so make sure one is available.
    if pop.num_links(sp, i) == 0:
        add_random_link(pop, sp, i)

    # Select a random link to be split in two with the new node between the two
    # halves.
    l = rand_range(0, pop.num_links(sp, i))
    old_link = pop.get_link(sp, i, l)

    # Make a node to go between the nodes that link connected, and allocate a
    # variable for its index (to be determined below).
    node = make_random_node(NodeKinds.HIDDEN.value)
    n = 0

    # Pick a place to put this node, someplace after the from_node and
    # before the to_node (though we might take its place). To respect the
    # node order invariant, the new index must be between the last input
    # node and the first output node.
    num_inputs, num_outputs = ti.static(pop.network_shape)
    n = rand_range(
        ti.max(old_link.from_node, num_inputs - 1),
        ti.min(old_link.to_node, pop.num_nodes(sp, i) - num_outputs)
    ) + 1
    pop.insert_node(sp, i, n, node)

    # Insert may have updated node indexes, so refresh our local copy of this
    # link to make sure from_node and to_node are up to date.
    old_link = pop.get_link(sp, i, l)
    # Add new links (with new innovation numbers) linking the old from_node to
    # the new node (with index n) to the old to_node.
    pop.add_link(sp, i, Link(old_link.from_node, n, old_link.weight))
    pop.add_link(sp, i, Link(n, old_link.to_node, 1.0))

    # Delete the link that we removed.
    pop.delete_link(sp, i, l)


# TODO: Optimize? This is inherently inefficient, since different threads are
# doing completely different work, but it may be possible to speed up by
# reducing the length of the longest branch or by reorienting the computation
# so we apply the same mutation to many genes at once.
@ti.func
def mutate_one(pop, sp, i):
    """Randomly choose one kind of mutation and apply it."""
    mutation_kind = rand_range(0, 8)

    # Add node
    # NOTE: no mutation is applied if the CPPN is already at max size.
    if mutation_kind == 0 and pop.has_room_for(sp, i, nodes=1, links=2):
        add_random_node(pop, sp, i)

    # Remove node
    # NOTE: no mutation is applied if the CPPN has no hidden nodes.
    elif mutation_kind == 1:
        num_inputs, num_outputs = ti.static(pop.network_shape)
        num_hidden = pop.num_nodes(sp, i) - num_inputs - num_outputs
        if num_hidden > 0:
            n = rand_range(num_inputs, num_inputs + num_hidden)
            pop.delete_node(sp, i, n)

    # Change activation
    elif mutation_kind == 2:
        n = rand_range(0, pop.num_nodes(sp, i))
        node = pop.get_node(sp, i, n)
        node.act_func = activation_funcs.random()
        pop.set_node(sp, i, n, node)

    # Change bias
    elif mutation_kind == 3:
        n = rand_range(0, pop.num_nodes(sp, i))
        node = pop.get_node(sp, i, n)
        node.bias = ti.math.clamp(
            node.bias + ti.randn() * BIAS_RANGE,
            -BIAS_RANGE, BIAS_RANGE)
        pop.set_node(sp, i, n, node)

    # Change gain
    elif mutation_kind == 3:
        n = rand_range(0, pop.num_nodes(sp, i))
        node = pop.get_node(sp, i, n)
        node.gain = ti.math.clamp(
            node.gain + ti.randn() *  GAIN_RANGE,
            -GAIN_RANGE, GAIN_RANGE)
        pop.set_node(sp, i, n, node)

    # Add link
    # NOTE: no mutation is applied if the CPPN is already at max size.
    elif mutation_kind == 5 and pop.has_room_for(sp, i, nodes=0, links=1):
        add_random_link(pop, sp, i)

    # Remove link
    # NOTE: no mutation is applied if the CPPN has no links.
    elif mutation_kind == 6 and pop.num_links(sp, i) > 0:
        l = rand_range(0, pop.num_links(sp, i))
        pop.delete_link(sp, i, l)

    # Change weight
    elif mutation_kind == 7:
        if pop.num_links(sp, i) > 0:
            l = rand_range(0, pop.num_links(sp, i))
            link = pop.get_link(sp, i, l)
            link.weight = ti.math.clamp(
                link.weight + ti.randn() * WEIGHT_RANGE,
                -WEIGHT_RANGE, WEIGHT_RANGE)
            pop.set_link(sp, i, l, link)
        # If there are no links, add a random link instead instead of not
        # applying any mutation.
        else:
            add_random_link(pop, sp, i)


@ti.func
def get_mate_link(pop, matches, sp, i, pl):
    p, m = matches[sp, i]
    ml = NONE
    # Grab the indicated link from parent, then go through all links in mate
    # and if one has the same innovation number, that's the result.
    p_link = pop.get_link(sp, p, pl)
    for l in range(pop.num_links(sp, m)):
        m_link = pop.get_link(sp, m, l)
        if p_link.innov == m_link.innov:
            ml = l
            break
    return ml


@ti.func
def is_compatible(pop, sp, i, matches):
    """Returns True iff parent and mate are similar enough to breed."""
    num_disjoint = 0
    num_total = 0
    weight_delta = 0.0

    # TODO: Note that we compute this currently is asymmetrical. The
    # compatibility of (parent, mate) != (mate, parent). This will change soon
    # when I redesign this to compute compatibility for all pairs.
    p, m = matches[sp, i]
    for pl in range(pop.num_links(sp, p)):
        ml = get_mate_link(pop, matches, sp, i, pl)
        # If both parent and mate have this link, compare their weights.
        if ml != NONE:
            num_total += 1
            weight_delta += ti.abs(
                pop.get_link(sp, p, pl).weight -
                pop.get_link(sp, m, ml).weight)

        # If just one of parent and mate have this link, it's "disjoint." The
        # original NEAT algorithm discriminates between "disjoint" and "excess"
        # genes, but here we treat them all the same.
        else:
            num_total += 1
            num_disjoint += 1

    return ((DISJOINT_COEFF * num_disjoint / num_total) +
            (WEIGHT_COEFF * weight_delta)) < COMPATIBILITY_THRESHOLD


@ti.func
def should_crossover(pop, sp, i, matches):
    p, m = matches[sp, i]
    # Reproduction is asexual by default, but if selection chose a distinct
    # parent and mate, they are compatible with each other, and luck is on
    # their side, then perform sexual reproduction with crossover.
    result = False
    if p != m and ti.random() < CROSSOVER_RATE:
        result = is_compatible(pop, sp, i, matches)
    return result


@ti.func
def crossover(pop, sp, i, matches):
    b_out = 1 - pop.buffer_index[None]
    p, m = matches[sp, i]

    # Copy nodes from parent to child. Note that if mate has nodes that parent
    # doesn't, they won't be copied. This is safe because we only take links
    # from mate that have corresponding links in parent, which means they will
    # never refer to a node that parent doesn't have.
    num_nodes = pop.num_nodes(sp, p)
    pop.node_lens[b_out, sp, i] = num_nodes
    for n in range(num_nodes):
        pop.nodes[b_out, sp, i, n] = pop.get_node(sp, p, n)

    # Fill in links for this individual by taking them from the parent.
    num_links = pop.num_links(sp, p)
    pop.link_lens[b_out, sp, i] = num_links
    for l in range(num_links):
        pl = l
        link = pop.get_link(sp, p, pl)

        # If the mate has a corresponding link, flip a coin to decide which
        # link weight to use.
        ml = get_mate_link(pop, matches, sp, i, pl)
        if ml != NONE and ti.random(dtype=int) % 2:
            link.weight = pop.get_link(sp, m, ml).weight

        # Fill in this link, using the copy from either parent or mate.
        pop.links[b_out, sp, i, l] = link


@ti.func
def clone(pop, sp, i, matches):
    b_out = 1 - pop.buffer_index[None]
    p, _ = matches[sp, i]

    num_nodes = pop.num_nodes(sp, p)
    pop.node_lens[b_out, sp, i] = num_nodes
    for n in range(num_nodes):
        pop.nodes[b_out, sp, i, n] = pop.get_node(sp, p, n)

    num_links = pop.num_links(sp, p)
    pop.link_lens[b_out, sp, i] = num_links
    for l in range(num_links):
        pop.links[b_out, sp, i, l] = pop.get_link(sp, p, l)


@ti.kernel
def propagate(pop: ti.template(), matches: ti.template()):
    # Generate each individual in the new population, generate one from the old
    # population, either by cloning or crossover.
    for sp, i in ti.ndrange(*pop.population_shape):
        if should_crossover(pop, sp, i, matches):
            crossover(pop, sp, i, matches)
        else:
            clone(pop, sp, i, matches)
    # After this function is completed, the population object will rotate its
    # double buffer and call mutate_all().


@ti.kernel
def mutate_all(pop: ti.template()):
    for sp, i in ti.ndrange(*pop.population_shape):
        if ti.random() < MUTATION_RATE:
            mutate_one(pop, sp, i)


@ti.kernel
def random_init(pop: ti.template()):
    # Generate a population of individuals with no hidden nodes, random
    # activation functions, and one random mutation each.
    for sp, i in ti.ndrange(*pop.population_shape):
        num_inputs, num_outputs = ti.static(pop.network_shape)
        for n in range(num_inputs):
            pop.insert_node(sp, i, n, make_random_node(NodeKinds.INPUT.value))
        for n in range(num_inputs, num_inputs + num_outputs):
            pop.insert_node(sp, i, n, make_random_node(NodeKinds.OUTPUT.value))
        # Add a randomized number of mutations to make the initial population
        # somewhat diverse.
        for _ in range(rand_range(1, 9)):
            mutate_one(pop, sp, i)
