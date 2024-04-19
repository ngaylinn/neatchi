from enum import Enum

import taichi as ti

from .activation_funcs import ActivationFuncs

# TODO: Currently, this library does no clean up of these data structures as
# they evolve. Over time, deleted node and link objects will accumulate,
# wasting space and clock cycles, for no reason. It's unclear whether it is
# better to accept this mess, or to put in the time and effort to actively
# rebuild the data structures from time to time.

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
