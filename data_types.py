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

MAX_NETWORK_SIZE = 200

from .activation_funcs import ActivationFuncs

class NodeKinds(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


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


cppn_dtype = np.dtype([
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
    dtype=cppn_dtype
)
