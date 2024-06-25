import numpy as np
import taichi as ti

from .data_types import NodeKinds
from .activation_funcs import ActivationFuncs, NUM_ACTIVATION_FUNCS
from .actuators import MAX_NETWORK_SIZE
from .reproduction import BIAS_RANGE, GAIN_RANGE, WEIGHT_RANGE


def print_cppn(cppn):
    def print_edge(num_cells, top):
        print('┏' if top else '┗', end='')
        for cell in range(num_cells):
            if cell > 0:
                print('┳' if top else '┻', end='')
            print('━' * 8, end='')
        print('┓' if top else '┛')

    def print_body(cell_strs):
        print('┃' + '┃'.join(cell_strs) + '┃')

    num_nodes = len(cppn['nodes']['kind'])
    print_edge(num_nodes, top=True)
    print_body([f'{"Node " + str(n):^8}' for n in range(num_nodes)])
    print_body([f'{NodeKinds(cppn["nodes"]["kind"][n]).name:^8}'
                for n in range(num_nodes)])
    print_body([f'{ActivationFuncs(cppn["nodes"]["act_func"][n]).name:^8}'
                for n in range(num_nodes)])
    print_body([f'b={cppn["nodes"]["bias"][n]:+5.3f}'
                for n in range(num_nodes)])
    print_body([f'g={cppn["nodes"]["gain"][n]:+5.3f}'
                for n in range(num_nodes)])
    print_edge(num_nodes, top=False)

    num_links = len(cppn['links']['weight'])
    if num_links == 0:
        return
    print_edge(num_links, top=True)
    print_body([f'{"Link " + str(l):^8}' for l in range(num_links)])
    print_body([f'{cppn["links"]["from_node"][l]:>2} -> '
                f'{cppn["links"]["to_node"][l]:<2}'
                for l in range(num_links)])
    print_body([f'w={cppn["links"]["weight"][l]:+5.3f}'
                for l in range(num_links)])
    print_body([f'{cppn["links"]["innov"][l]:^8}'
                for l in range(num_links)])
    print_edge(num_links, top=False)


@ti.kernel
def validate(pop: ti.template(), sp: int, i: int) -> bool:
    """Check all invariants on the given NN, for debugging code in this file.

    Recommend running this with ti.init(arch=ti.cpu, cpu_max_num_threads=1) to
    ensure the last message on screen corresponds to the error that caused the
    assert to fail. Uncommon print statements below to see which node / link in
    the population triggered the error and why.
    """
    num_nodes = pop.num_nodes(sp, i)
    num_links = pop.num_links(sp, i)
    num_inputs, num_outputs = ti.static(pop.network_shape)
    valid = True
    if num_nodes < 0 or num_nodes >= MAX_NETWORK_SIZE:
        print(f'Expected num_nodes 0 <= ({num_nodes}) < MAX_NETWORK_SIZE')
        valid = False
    if num_links < 0 or num_links >= MAX_NETWORK_SIZE:
        print(f'Expected num_links 0 <= ({num_links}) < MAX_NETWORK_SIZE')
        valid = False
    for n in range(num_nodes):
        node = pop.get_node(sp, i, n)
        if n < num_inputs:
            if node.kind != NodeKinds.INPUT.value:
                print(f'The first {num_inputs} nodes should have type INPUT')
                valid = False
        elif n < num_nodes - num_outputs:
            if node.kind != NodeKinds.HIDDEN.value:
                print(f'Node {n} should have type HIDDEN')
                valid = False
        else:
            if node.kind != NodeKinds.OUTPUT.value:
                print(f'The last {num_outputs} nodes should have type OUTPUT')
                valid = False
            if node.act_func < 0 or node.act_func >= NUM_ACTIVATION_FUNCS:
                print(f'Node {n} has an invalid activation function')
                valid = False
            if node.bias < -BIAS_RANGE or node.bias > BIAS_RANGE:
                print(f'Node {n} has bias outside valid range ({-BIAS_RANGE} - {BIAS_RANGE})')
                valid = False
            if node.gain < -GAIN_RANGE or node.gain > GAIN_RANGE:
                print(f'Node {n} has gain outside valid range ({-GAIN_RANGE} - {GAIN_RANGE})')
                valid = False
    for l in range(num_links):
        link = pop.get_link(sp, i, l)
        if link.from_node < 0 or link.from_node >= num_nodes:
            print(f'Link {l} has from_node out of range')
            valid = False
        if link.to_node < num_inputs:
            print(f'Link {l} cannot have an INPUT to_node')
            valid = False
        if link.to_node >= num_nodes:
            print(f'Link {l} has to_node out of range')
            valid = False
        if link.to_node <= link.from_node:
            print(f'Link {l} references nodes in non-ascending order')
            valid = False
        if link.weight < -WEIGHT_RANGE or link.weight > WEIGHT_RANGE:
            print(f'Link {l} has weight outside valid range ({-WEIGHT_RANGE} - {WEIGHT_RANGE})')
            valid = False
        if link.innov >= pop.innovation_counter[None]:
            print(f'Link {l} has an invalid innovation number')
            valid = False
        for l2 in range(num_links):
            if l != l2:
                if link.innov == pop.get_link(sp, i, l2).innov:
                    print(f'Links {l} and {l2} have the same innovation number')
                    valid = False
    return valid


def validate_all(pop):
    for sp, i in np.ndindex(pop.population_shape):
        if not validate(pop, sp, i):
            cppn = pop.get_cppns([(sp, i)])[0]
            print(f'While validating individual ({sp}, {i}):')
            print_cppn(cppn)
            exit()
