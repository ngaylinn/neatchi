"""DataStructures representing CPPNs.

The Node class represents a single neuron and the Link class represents a
synapse between two neurons. Both data types are stored in dynamic fields,
which are "lists" semantically, but don't support removing items. For this
reason, both Nodes and Links have a deleted attribute, indicating they have
been deleted and should be ignored for activations. Reproduction.py does some
opportunistic cleanup of these data structures, but no effort is made to
regularly clear out all deleted nodes.
"""

from enum import Enum

import numpy as np
import taichi as ti

# In this library, each CPPN can have no more than this many nodes and this
# many links. This value effects the complexity of networks that can be
# evolved. It also affects memory and performance in two key ways. First, this
# determines the overall size of the CPPNs, since we always allocate enough
# space for this many nodes and links. This affects the time it takes to
# transmit CPPNs from GPU to CPU and back. Also, the Actuator class takes
# advantage of this value being small to hold the entire activation buffer in
# thread-local memory. This is a big performance boost, but won't work if this
# number gets much bigger. We'd have to switch to a field instead.
MAX_NETWORK_SIZE = 20


class NodeKinds(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


# NOTE: For large populations, the total memory footprint for the CPPNs can
# grow quite large, which means copying populations from the GPU can be quite
# slow. Generally, this should be avoided when possible. It should be possible
# to mitigate this cost by using quantized datatypes, but that would require
# significant refactoring.
@ti.dataclass
class Node:
    kind: ti.int8
    act_func: ti.int8
    # TODO: Does having bias and gain help?
    bias: float
    gain: float


@ti.dataclass
class Link:
    from_node: ti.int8
    to_node: ti.int8
    weight: float
    innov: int


CPPN_DTYPE = np.dtype([
    ('nodes', np.dtype([
        ('kind', np.int8),
        ('act_func', np.int8),
        ('bias', np.float32),
        ('gain', np.float32)
    ]), MAX_NETWORK_SIZE),
    ('links', np.dtype([
        ('from_node', np.int8),
        ('to_node', np.int8),
        ('weight', np.float32),
        ('innov', np.int32)
    ]), MAX_NETWORK_SIZE)
])


EMPTY_CPPN = np.array(
    [([(-1, -1, np.nan, np.nan)] * MAX_NETWORK_SIZE,
      [(-1, -1, np.nan, -1)] * MAX_NETWORK_SIZE)],
    dtype=CPPN_DTYPE
)


def import_cppns(pop, cppns):
    # Takes an array and expands it into a double-buffered version of
    # itself by copying it once into a new first axis.
    def double_buff(arr):
        return np.repeat(np.expand_dims(arr, 0), 2, 0)

    # Make sure the input data fits.
    assert cppns.shape == pop.population_shape
    assert cppns.dtype == CPPN_DTYPE

    pop.nodes.from_numpy(double_buff(cppns['nodes']))
    pop.node_lens.from_numpy(
        double_buff(np.count_nonzero(
            cppns['nodes']['kind'] >= 0, axis=2
        ).astype(np.int32)))

    pop.links.from_numpy(double_buff(cppns['links']))
    pop.link_lens.from_numpy(
        double_buff(np.count_nonzero(
            cppns['links']['innov'] >= 0, axis=2
        ).astype(np.int32)))


def export_cppns(pop):
    b = pop.buffer_index[None]
    nodes = pop.nodes.to_numpy()
    links = pop.links.to_numpy()
    result = np.full(pop.population_shape, EMPTY_CPPN)
    for sp, i in ti.ndrange(*pop.population_shape):
        num_nodes = pop.node_lens[b, sp, i]
        num_links = pop.link_lens[b, sp, i]
        for key, data in nodes.items():
            result[sp, i]['nodes'][key][:num_nodes] = \
                    data[b, sp, i, :num_nodes]
        for key, data in links.items():
            result[sp, i]['links'][key][:num_links] = \
                    data[b, sp, i, :num_links]
    return result
