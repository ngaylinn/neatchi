import taichi as ti

import activation
import activation_funcs
from data_types import Link, Node, NodeKinds, node_to_str, link_to_str
import reproduction

MUTATION_RATE = 0.01


@ti.data_oriented
class Population:
    def __init__(self, num_inputs=2, num_outputs=3, count=1):
        self.num_inputs = num_inputs
        self.input_type = ti.types.vector(num_inputs, float)
        self.num_outputs = num_outputs
        self.output_type = ti.types.vector(num_outputs, float)
        self.count = count

        # TODO: Tune sizes?
        n = ti.root.dense(ti.i, count).dynamic(ti.j, 1024, chunk_size=32)
        self.nodes = Node.field()
        n.place(self.nodes)

        # TODO: Tune sizes?
        l = ti.root.dense(ti.i, count).dynamic(ti.j, 1024, chunk_size=32)
        self.links = Link.field()
        l.place(self.links)

        # Populate the node list with inputs and outputs.
        self.init_kernel()

    @ti.kernel
    def init_kernel(self):
        for c in range(self.count):
            for _ in range(self.num_inputs):
                self.nodes[c].append(
                    Node(NodeKinds.INPUT.value,
                         activation_funcs.random()))
            for _ in range(self.num_outputs):
                self.nodes[c].append(
                    Node(NodeKinds.OUTPUT.value,
                         activation_funcs.random()))

    @ti.kernel
    def num_nodes(self, c: int) -> int:
        return self.nodes[c].length()

    @ti.kernel
    def num_links(self, c: int) -> int:
        return self.links[c].length()

    def print(self, c):
        def print_edge(num_cells, top):
            print('┏' if top else '┗', end='')
            for cell in range(num_cells):
                if cell > 0:
                    print('┳' if top else '┻', end='')
                print('━' * 8, end='')
            print('┓' if top else '┛')

        def print_body(row_strs):
            for row in range(len(row_strs[0])):
                for cell in range(len(row_strs)):
                    print('┃' + row_strs[cell][row], end='')
                print('┃')

        num_nodes = self.num_nodes(c)
        print_edge(num_nodes, top=True)
        print_body([node_to_str(self.nodes[c, n]).split('\n')
                    for n in range(num_nodes)])
        print_edge(num_nodes, top=False)

        num_links = self.num_links(c)
        if num_links == 0:
            return
        print_edge(num_links, top=True)
        print_body([link_to_str(self.links[c, n]).split('\n')
                    for n in range(num_links)])
        print_edge(num_links, top=False)

    def mutate(self):
        return reproduction.mutate(self)

    def crossover(self):
        return reproduction.crossover(self)

    def activate(self, inputs, outputs):
        return activation.activate(self, inputs, outputs)

    def render(self, image_field):
        return activation.render(self, image_field)
