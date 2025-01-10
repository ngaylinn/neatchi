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

from .data_types import Node, NodeKinds, Link

MAX_INITIAL_MUTATIONS = 8
MUTATION_RATE = 0.01

NONE = -1

# TODO: Are these ranges reasonable?
BIAS_RANGE = 1.0
GAIN_RANGE = 8.0
WEIGHT_RANGE = 1.0

@ti.func
def rand_range(min_val, max_val):
    """Pick a random int from the given range."""
    return ti.random(dtype=int) % (max_val - min_val) + min_val


@ti.func
def add_random_link(pop, sp, i):
    """Add a random link to the specified CPPN as a new innovation."""
    num_nodes = pop.num_nodes(sp, i)

    from_node, to_node = 0, 0
    # Make sure from_node < to_node and disallow links from output nodes and to
    # input nodes.
    from_node = ti.cast(rand_range(0, num_nodes - pop.num_outputs), ti.int8)
    to_node = ti.cast(
        rand_range(ti.max(from_node + 1, pop.num_inputs), num_nodes), ti.int8)

    # Actually make the link, and make sure it gets an innovation number.
    pop.add_link(sp, i, Link(from_node, to_node, ti.random()))


@ti.func
def make_random_node(node_type):
    return Node(
        ti.cast(node_type, ti.int8),
        ti.cast(ti.random(dtype=int), ti.int8),
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
    n = ti.cast(rand_range(
        ti.max(old_link.from_node, pop.num_inputs - 1),
        ti.min(old_link.to_node, pop.num_nodes(sp, i) - pop.num_outputs)
    ) + 1, ti.int8)
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
        num_hidden = pop.num_nodes(sp, i) - pop.num_inputs - pop.num_outputs
        if num_hidden > 0:
            n = rand_range(pop.num_inputs, pop.num_inputs + num_hidden)
            pop.delete_node(sp, i, n)

    # Change activation
    elif mutation_kind == 2:
        n = rand_range(0, pop.num_nodes(sp, i))
        node = pop.get_node(sp, i, n)
        node.act_func = ti.cast(ti.random(dtype=int), ti.int8)
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
def get_mate_link(pop, sp, p, pl, m):
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
def crossover(pop, sp, i, p, m):
    b_out = 1 - pop.buffer_index[None]

    # Initialize this individual's node list from the parent. If the mate has
    # nodes the parent doesn't, they won't be copied. This is safe because we
    # only take links from mate that have corresponding links in parent, which
    # means they will never refer to a node that parent doesn't have. We may
    # copy nodes from mate but that happens when we traverse the link list,
    # because that's the only way to know which nodes correspond to each other.
    num_nodes = pop.num_nodes(sp, p)
    pop.node_lens[b_out, sp, i] = num_nodes
    for n in range(num_nodes):
        pop.nodes[b_out, sp, i, n] = pop.get_node(sp, p, n)

    # Fill in links for this individual by taking them from the parent.
    num_links = pop.num_links(sp, p)
    pop.link_lens[b_out, sp, i] = num_links
    for l in range(num_links):
        pl = l
        p_link = pop.get_link(sp, p, pl)
        link = p_link

        # If the mate has a corresponding link, flip a coin to decide whether
        # we should crossover that link and its corresponding to_node.
        ml = get_mate_link(pop, sp, p, pl, m)
        if ml != NONE and ti.random(dtype=int) % 2:
            # Copy the link weight from mate to parent. Node indices may not
            # match, so leave those unchanged.
            m_link = pop.get_link(sp, m, ml)
            link.weight = m_link.weight

            # Also copy over the associated to_node. If the links have the same
            # innovation number, then their associated nodes should also be
            # corresponding, even though they may have different indices.
            # TODO: Should I also copy the from_node? Feels a little arbitrary.
            pn = int(p_link.to_node)
            mn = int(m_link.to_node)
            pop.nodes[b_out, sp, i, pn] = pop.get_node(sp, m, mn)

        # Fill in this link, using the copy from either parent or mate.
        pop.links[b_out, sp, i, l] = link


@ti.func
def clone(pop, sp, i, p):
    b_out = 1 - pop.buffer_index[None]

    num_nodes = pop.num_nodes(sp, p)
    pop.node_lens[b_out, sp, i] = num_nodes
    for n in range(num_nodes):
        pop.nodes[b_out, sp, i, n] = pop.get_node(sp, p, n)

    num_links = pop.num_links(sp, p)
    pop.link_lens[b_out, sp, i] = num_links
    for l in range(num_links):
        pop.links[b_out, sp, i, l] = pop.get_link(sp, p, l)


@ti.kernel
def propagate(pop: ti.template(), g: int):
    # Generate each individual in each new sub population, generate one from
    # the old sub population, either by cloning or crossover.
    for sp, i in ti.ndrange(*pop.population_shape):
        p, m = pop.matchmaker.matches[g, sp, i]
        if m == NONE:
            clone(pop, sp, i, p)
        else:
            crossover(pop, sp, i, p, m)
    # After this function is completed, the population object will rotate its
    # double buffer and call mutate_all().


@ti.kernel
def mutate_all(pop: ti.template(), mutation_rate: float):
    for sp, i in ti.ndrange(*pop.population_shape):
        # Do a number of mutations proportional to the size of the CPPN.
        # Without this, the chance that an individual node or link will be
        # mutated goes down as the CPPN evolves more complexity, which is
        # undesirable. Note, this likely increases thread divergence and
        # therefore hurts performance.
        network_size = pop.num_nodes(sp, i) + pop.num_links(sp, i)
        for _ in range(network_size):
            if ti.random() < mutation_rate:
                mutate_one(pop, sp, i)


@ti.kernel
def random_init(pop: ti.template()):
    # Generate a population of individuals with no hidden nodes, random
    # activation functions, and one random mutation each.
    for sp, i in ti.ndrange(*pop.population_shape):
        for n in range(pop.num_inputs):
            pop.insert_node(sp, i, n, make_random_node(NodeKinds.INPUT.value))
        for n in range(pop.num_inputs, pop.num_inputs + pop.num_outputs):
            pop.insert_node(sp, i, n, make_random_node(NodeKinds.OUTPUT.value))

        # Add a randomized number of mutations to make the initial population
        # somewhat diverse. The number of mutations is proportional to the
        # initial network size, so that big networks are as randomized as small
        # ones.
        network_size = pop.num_inputs + pop.num_outputs
        min_mutations = network_size
        max_mutations = network_size * MAX_INITIAL_MUTATIONS + 1
        for _ in range(rand_range(min_mutations, max_mutations)):
            mutate_one(pop, sp, i)
