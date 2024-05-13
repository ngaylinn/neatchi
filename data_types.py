"""DataStructures representing CPPNs.

The Node class represents a single neuron and the Link class represents a
synapse between two neurons. Both data types are stored in dynamic fields,
which are logically "lists" but don't support removing items. For this reason,
both Nodes and Links have a deleted attribute, indicating they have been
deleted and should be ignored for activations. Reproduction.py does some
opportunistic cleanup of these data structures, but no effort is made to
regularly clear out all deleted nodes.
"""

from enum import Enum

import taichi as ti

from .activation_funcs import ActivationFuncs

class NodeKinds(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


@ti.dataclass
class Node:
    kind: int
    act_func: int
    bias: float  # TODO: Do we need bias? Do we also want gain?
    deleted: bool


def node_to_str(node):
    return (f'{NodeKinds(node.kind).name:^8}\n'
            f'{ActivationFuncs(node.act_func).name:^8}\n'
            f'b={node.bias:6.4f}\n'
            f'{"DELETED " if node.deleted else "        "}')


@ti.dataclass
class Link:
    from_node: int
    to_node: int
    weight: float
    deleted: bool
    innov: int


def link_to_str(link):
    from_to = f'{link.from_node} -> {link.to_node}'
    innov = f'i={link.innov}'
    return (f'{from_to:^8}\n'
            f'w={link.weight:6.4f}\n'
            f'{innov:^8}\n'
            f'{"DELETED " if link.deleted else "        "}')
