import taichi as ti

import activation_funcs
from data_types import Link, Node, NodeKinds

# TODO: Manage innovation numbers!
@ti.func
def add_random_link(pop, c):
    num_nodes = pop.nodes[c].length()
    # Randomly pick any node to be the "from" node for this link.
    n_from = ti.random(dtype=int) % num_nodes
    # Randomly pick a non-input node to be the "to" node for this link.
    n_to = ti.random(dtype=int) % (num_nodes - pop.num_inputs)
    n_to += pop.num_inputs
    pop.links[c].append(Link(n_from, n_to, ti.random()))


@ti.func
def mutate_one(pop, c):
    mutation_kind = ti.random(dtype=int) % 7
    # Add node
    # TODO: Manage innovation numbers!
    if mutation_kind == 0:
        if pop.links[c].length() == 0:
            add_random_link(pop, c)
        l = ti.random(dtype=int) % pop.links[c].length()
        weight = pop.links[c, l].weight
        from_node = pop.links[c, l].from_node
        to_node = pop.links[c, l].to_node
        pop.links[c, l].disabled = True
        n = pop.nodes[c].length()
        pop.nodes[c].append(
            Node(NodeKinds.HIDDEN.value,
                 activation_funcs.random(), ti.random()))
        pop.links[c].append(Link(from_node, n, weight))
        pop.links[c].append(Link(n, to_node, 1.0))

    # Remove node
    elif mutation_kind == 1:
        num_frozen = pop.num_inputs + pop.num_outputs
        num_hidden = pop.nodes[c].length() - num_frozen
        if num_hidden > 0:
            n = ti.random(dtype=int) % num_hidden + num_frozen
            pop.nodes[c, n].disabled = True
            for l in range(pop.links[c].length()):
                link = pop.links[c, l]
                if (link.from_node == n or link.to_node == n):
                    pop.links[c, l].disabled = True

    # Change activation
    elif mutation_kind == 2:
        n = ti.random(dtype=int) % pop.nodes[c].length()
        pop.nodes[c, n].act_func = activation_funcs.random()

    # Change bias
    elif mutation_kind == 3:
        n = ti.random(dtype=int) % pop.nodes[c].length()
        pop.nodes[c, n].bias = ti.random()

    # Add link
    elif mutation_kind == 4:
        add_random_link(pop, c)

    # Remove link
    elif mutation_kind == 5 and pop.links[c].length() > 0:
        l = ti.random(dtype=int) % pop.links[c].length()
        pop.links[c, l].disabled = True

    # Change weight
    elif mutation_kind == 6:
        if pop.links[c].length() > 0:
            l = ti.random(dtype=int) % pop.links[c].length()
            pop.links[c, l].weight = ti.random()
        else:
            add_random_link(pop, c)

@ti.kernel
def mutate(pop: ti.template()):
    for c in range(pop.count):
        mutate_one(pop, c)


def crossover(pop):
    ...
