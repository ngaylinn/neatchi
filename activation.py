import taichi as ti

import activation_funcs

@ti.func
def activate_one(cppns, c, inputs):
    num_nodes = cppns.nodes[c].length()
    num_links = cppns.links[c].length()

    # Populate input node activations
    for n in range(cppns.num_inputs):
        cppns.nodes[c, n].prev_act = inputs[n]

    # Preserve previous activations before computing the next round.
    # TODO: It'd be faster to swap buffer pointers than to copy.
    for n in range(cppns.num_inputs, num_nodes):
        cppns.nodes[c, n].prev_act = cppns.nodes[c, n].curr_act

    # Compute activations for non-input nodes
    for n in range(cppns.num_inputs, num_nodes):
        if not cppns.nodes[c, n].disabled:
            raw = 0.0
            for l in range(num_links):
                link = cppns.links[c, l]
                if not link.disabled and link.to_node == n:
                    value = cppns.nodes[c, link.from_node].prev_act
                    weight = link.weight
                    # TODO: support different aggregation funcs?
                    raw += value * weight
            next_act = activation_funcs.call(
                cppns.nodes[c, n].act_func,
                raw + cppns.nodes[c, n].bias)
            cppns.nodes[c, n].curr_act = next_act

    # Return just the activations from the output nodes
    return cppns.output_type(
        [cppns.nodes[c, n].curr_act
         for n in range(cppns.num_inputs,
                        cppns.num_inputs + cppns.num_outputs)])

@ti.kernel
def activate(cppns: ti.template(),
             inputs: ti.template(), outputs: ti.template()):
    for c in range(cppns.count):
        outputs[c] = activate_one(cppns, c, inputs[c])

@ti.kernel
def render_kernel(cppns: ti.template(), image_field: ti.template(),
                  x_dim: int, y_dim: int):
    for c, x, y in ti.ndrange(cppns.count, x_dim, y_dim):
        image_field[c, x, y] = activate_one(cppns, c, cppns.input_type(x, y))

def render(cppns, image_field):
    count, x_dim, y_dim = image_field.shape
    channels = image_field.n
    assert(count == cppns.count)
    # TODO: support higher dimensions in rendering?
    assert(cppns.num_inputs == 2)
    assert(channels == cppns.num_outputs)
    render_kernel(cppns, image_field, x_dim, y_dim)

