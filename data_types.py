from enum import Enum

import taichi as ti

from activation_funcs import ActivationFuncs


class NodeKinds(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


@ti.dataclass
class Node:
    kind: int
    act_func: int
    bias: float  # TODO: Do we need bias? Do we also want gain?
    disabled: bool
    prev_act: float
    curr_act: float


def node_to_str(node):
    return (f'{NodeKinds(node.kind).name:^8}\n'
            f'{ActivationFuncs(node.act_func).name:^8}\n'
            f'b={node.bias:6.4f}\n' +
            ('DISABLED' if node.disabled else 'ENABLED '))


@ti.dataclass
class Link:
    from_node: int
    to_node: int
    weight: float
    disabled: bool
    innov: int


def link_to_str(link):
    from_to = f'{link.from_node} -> {link.to_node}'
    innov = f'i={link.innov}'
    return (f'{from_to:^8}\n'
            f'w={link.weight:6.4f}\n'
            f'{innov:^8}\n' +
            ('DISABLED' if link.disabled else 'ENABLED '))


