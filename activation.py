"""Activation functions for the CPPNs.

In a CPPN, each neuron has its own activation function, chosen from the
ActivationFuncs enum. Use the random() function to get a randomly chosen
activation function and the call() function to invoke one by its enum value.
"""

from enum import Enum

import taichi as ti

from .data_types import MAX_NETWORK_SIZE

class ActivationFuncs(Enum):
    SIGMOID = 0
    TANH = 1
    SIN = 2
    GAUS = 3
    RELU = 4
    IDENTITY = 5
    CLAMPED = 6
    INV = 7
    LOG = 8
    EXP = 9
    ABS = 10
    SQUARE = 11
    CUBE = 12
    TRIANGLE = 13
    SAWTOOTH = 14
    SQR_WAVE = 15
    NOTCH = 16
    STEP = 17

NUM_ACTIVATION_FUNCS = len(ActivationFuncs.__members__)

# These functions have been tuned arbitrarily to have an "appropriate" range of
# values when applied to values in the range from 0.0 to 1.0.
@ti.func
def activate_sigmoid(pre_activation):
    return 1.0 / (1.0 + ti.exp(8 * pre_activation - 4))

@ti.func
def activate_tanh(pre_activation):
    return ti.tanh(8 * pre_activation - 4)

@ti.func
def activate_sin(pre_activation):
    return ti.sin(2 * pre_activation * ti.math.pi - ti.math.pi)

@ti.func
def activate_gaus(pre_activation):
    return ti.exp(-(4 * pre_activation - 2) ** 2)

@ti.func
def activate_relu(pre_activation):
    return ti.max(0, pre_activation)

@ti.func
def activate_identity(pre_activation):
    return pre_activation

@ti.func
def activate_clamped(pre_activation):
    return ti.math.clamp(pre_activation, 0.0, 1.0)

@ti.func
def activate_inv(pre_activation):
    return ti.select(pre_activation == 0, 0, 1.0 / pre_activation)

@ti.func
def activate_log(pre_activation):
    return ti.log(max(1e-7, 8 * pre_activation))

@ti.func
def activate_exp(pre_activation):
    return ti.exp(4 * pre_activation - 2)

@ti.func
def activate_abs(pre_activation):
    return ti.abs(pre_activation)

@ti.func
def activate_square(pre_activation):
    return pre_activation ** 2

@ti.func
def activate_cube(pre_activation):
    return pre_activation ** 3

@ti.func
def activate_triangle(pre_activation):
    return 2 * ti.abs(2 * (pre_activation - ti.floor(pre_activation + .5))) - 1

@ti.func
def activate_sawtooth(pre_activation):
    return 4 * ((pre_activation) % 0.5) - 1

@ti.func
def activate_sqr_wave(pre_activation):
    return 2 * ((pre_activation % 1) > 0.5) - 1

@ti.func
def activate_notch(pre_activation):
    return pre_activation > -0.1 and pre_activation < 0.1

@ti.func
def activate_step(pre_activation):
    return pre_activation // 1.0

@ti.func
def activate_node(node, raw):
    act_func = node.act_func % NUM_ACTIVATION_FUNCS
    pre_activation = node.gain * raw + node.bias
    post_activation = 0.0
    if act_func == ActivationFuncs.SIGMOID.value:
        post_activation = activate_sigmoid(pre_activation)
    elif act_func == ActivationFuncs.TANH.value:
        post_activation = activate_tanh(pre_activation)
    elif act_func == ActivationFuncs.SIN.value:
        post_activation = activate_sin(pre_activation)
    elif act_func == ActivationFuncs.GAUS.value:
        post_activation = activate_gaus(pre_activation)
    elif act_func == ActivationFuncs.RELU.value:
        post_activation = activate_relu(pre_activation)
    elif act_func == ActivationFuncs.IDENTITY.value:
        post_activation = activate_identity(pre_activation)
    elif act_func == ActivationFuncs.CLAMPED.value:
        post_activation = activate_clamped(pre_activation)
    elif act_func == ActivationFuncs.INV.value:
        post_activation = activate_inv(pre_activation)
    elif act_func == ActivationFuncs.LOG.value:
        post_activation = activate_log(pre_activation)
    elif act_func == ActivationFuncs.EXP.value:
        post_activation = activate_exp(pre_activation)
    elif act_func == ActivationFuncs.ABS.value:
        post_activation = activate_abs(pre_activation)
    elif act_func == ActivationFuncs.SQUARE.value:
        post_activation = activate_square(pre_activation)
    elif act_func == ActivationFuncs.CUBE.value:
        post_activation = activate_cube(pre_activation)
    elif act_func == ActivationFuncs.TRIANGLE.value:
        post_activation = activate_triangle(pre_activation)
    elif act_func == ActivationFuncs.SAWTOOTH.value:
        post_activation = activate_sawtooth(pre_activation)
    elif act_func == ActivationFuncs.SQR_WAVE.value:
        post_activation = activate_sqr_wave(pre_activation)
    elif act_func == ActivationFuncs.NOTCH.value:
        post_activation = activate_notch(pre_activation)
    elif act_func == ActivationFuncs.STEP.value:
        post_activation = activate_step(pre_activation)
    return post_activation


@ti.func
def activate_network(pop, w, inputs):
    act = ti.Vector([0.0] * MAX_NETWORK_SIZE)
    sp, i = pop.index[w]
    num_nodes = pop.num_nodes(sp, i)
    num_links = pop.num_links(sp, i)

    # Put input values into the activation buffer.
    for n in range(pop.num_inputs):
        act[n] = inputs[n]

    # Activate the CPPN. This network is a DAG with nodes sorted by depth,
    # so we can just traverse them in order and compute their final value.
    for n in range(pop.num_inputs, num_nodes):
        raw = 0.0
        for l in range(num_links):
            link = pop.get_link(sp, i, l)
            if link.to_node == n:
                value = act[link.from_node]
                raw += value * link.weight
        node = pop.get_node(sp, i, n)
        act[n] = activate_node(node, raw)

    # Capture the values from the CPPN's output nodes and return them.
    outputs = ti.Vector([0.0] * pop.num_outputs)
    for o in range(pop.num_outputs):
        n = num_nodes - pop.num_outputs + o
        outputs[o] = ti.math.clamp(act[n], 0.0, 1.0)
    return outputs


# -----------------------------------------------------------------------------
# For debugging: render a small test swatch for each activation function.
# -----------------------------------------------------------------------------

@ti.kernel
def render_all(field: ti.template()):
    for a, r, c in ti.ndrange(NUM_ACTIVATION_FUNCS, 64, 64):
        ar = (a % 5) * 65 + r
        ac = (a // 5) * 65 + c
        field[ar, ac] = ti.math.clamp(0.5 * call(a, c / 64) + 0.5, 0.0, 1.0)


if __name__ == '__main__':
    ti.init()
    extent = 5*65-1
    shape = (extent, extent)
    field = ti.field(float, shape=shape)
    render_all(field)
    gui = ti.GUI('Test', shape)
    while gui.running:
        gui.set_image(field)
        for l in range(1, 5):
            gui.line((0, l*65/extent), (extent, l*65/extent), color=0xFF00FF)
            gui.line((l*65/extent, 0), (l*65/extent, extent), color=0xFF00FF)
        for a in range(NUM_ACTIVATION_FUNCS):
            x = (a % 5) * 65 / extent
            y = ((a // 5)+1) * 65 / extent
            gui.text(str(a), (x, y), color=0xFF00FF)
        gui.show()
