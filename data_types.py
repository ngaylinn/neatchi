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


# TODO: Would it be better to use quantized data types? Most values here only
# use a small range, and data transfer speed over the PCI bus does seem to be a
# limiting factor in some applications. However, activation speed is also
# critical, so we'll want to ensure the overhead of packing / unpacking data
# doesn't slow things down too much.
@ti.dataclass
class Node:
    kind: int
    act_func: int
    # TODO: Does having bias and gain help?
    bias: float
    gain: float


@ti.dataclass
class Link:
    from_node: int
    to_node: int
    weight: float
    innov: int


CPPN_DTYPE = np.dtype([
    ('nodes', np.dtype([
        ('kind', np.int32),
        ('act_func', np.int32),
        ('bias', np.float32),
        ('gain', np.float32)
    ]), MAX_NETWORK_SIZE),
    ('links', np.dtype([
        ('from_node', np.int32),
        ('to_node', np.int32),
        ('weight', np.float32),
        ('innov', np.int32)
    ]), MAX_NETWORK_SIZE)
])


EMPTY_CPPN = np.array(
    [([(-1, -1, np.nan, np.nan)] * MAX_NETWORK_SIZE,
      [(-1, -1, np.nan, -1)] * MAX_NETWORK_SIZE)],
    dtype=CPPN_DTYPE
)
