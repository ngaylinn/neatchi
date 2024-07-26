"""Activation functions for the CPPNs.

In a CPPN, each neuron has its own activation function, chosen from the
ActivationFuncs enum. Use the random() function to get a randomly chosen
activation function and the call() function to invoke one by its enum value.
"""

from enum import Enum

import taichi as ti

from .data_types import MAX_NETWORK_SIZE


# These functions have been tuned arbitrarily to have an "appropriate" range of
# values when applied to values in the range from 0.0 to 1.0.
@ti.func
def activate_sigmoid(raw):
    return 1.0 / (1.0 + ti.exp(8 * raw - 4))

@ti.func
def activate_tanh(raw):
    return ti.tanh(8 * raw - 4)

@ti.func
def activate_sin(raw):
    return ti.sin(2 * raw * ti.math.pi - ti.math.pi)

@ti.func
def activate_gaus(raw):
    return ti.exp(-(4 * raw - 2) ** 2)

@ti.func
def activate_relu(raw):
    return ti.max(0, raw)

@ti.func
def activate_identity(raw):
    return raw

@ti.func
def activate_clamped(raw):
    return ti.math.clamp(raw, 0.0, 1.0)

@ti.func
def activate_inv(raw):
    return ti.select(raw == 0, 0, 1.0 / raw)

@ti.func
def activate_log(raw):
    return ti.log(max(1e-7, 8 * raw))

@ti.func
def activate_exp(raw):
    return ti.exp(4 * raw - 2)

@ti.func
def activate_abs(raw):
    return ti.abs(raw)

@ti.func
def activate_square(raw):
    return raw ** 2

@ti.func
def activate_cube(raw):
    return raw ** 3

@ti.func
def activate_triangle(raw):
    return 2 * ti.abs(2 * (raw - ti.floor(raw + .5))) - 1

@ti.func
def activate_sawtooth(raw):
    return 4 * ((raw) % 0.5) - 1

@ti.func
def activate_sqr_wave(raw):
    return 2 * ((raw % 1) > 0.5) - 1

@ti.func
def activate_notch(raw):
    return raw > -0.1 and raw < 0.1

@ti.func
def activate_step(raw):
    return raw // 1.0


@ti.data_oriented
class ActivationFuncs:
    class Keys(Enum):
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

    count = len(Keys.__members__)

    @ti.func
    def random(self):
        return ti.cast(
            ti.random(dtype=int) % ti.static(ActivationFuncs.count),
            ti.int8)

    @staticmethod
    def name(key):
        return ActivationFuncs.Keys(key).name

    # NOTE: These inputs provided temporarily for override support in
    # experimental code. This will likely change in the near future.
    @ti.func
    def call(self, inputs, w, n, key, raw):
        value = 0.0
        if key == ActivationFuncs.Keys.SIGMOID.value:
            value = activate_sigmoid(raw)
        elif key == ActivationFuncs.Keys.TANH.value:
            value = activate_tanh(raw)
        elif key == ActivationFuncs.Keys.SIN.value:
            value = activate_sin(raw)
        elif key == ActivationFuncs.Keys.GAUS.value:
            value = activate_gaus(raw)
        elif key == ActivationFuncs.Keys.RELU.value:
            value = activate_relu(raw)
        elif key == ActivationFuncs.Keys.IDENTITY.value:
            value = activate_identity(raw)
        elif key == ActivationFuncs.Keys.CLAMPED.value:
            value = activate_clamped(raw)
        elif key == ActivationFuncs.Keys.INV.value:
            value = activate_inv(raw)
        elif key == ActivationFuncs.Keys.LOG.value:
            value = activate_log(raw)
        elif key == ActivationFuncs.Keys.EXP.value:
            value = activate_exp(raw)
        elif key == ActivationFuncs.Keys.ABS.value:
            value = activate_abs(raw)
        elif key == ActivationFuncs.Keys.SQUARE.value:
            value = activate_square(raw)
        elif key == ActivationFuncs.Keys.CUBE.value:
            value = activate_cube(raw)
        elif key == ActivationFuncs.Keys.TRIANGLE.value:
            value = activate_triangle(raw)
        elif key == ActivationFuncs.Keys.SAWTOOTH.value:
            value = activate_sawtooth(raw)
        elif key == ActivationFuncs.Keys.SQR_WAVE.value:
            value = activate_sqr_wave(raw)
        elif key == ActivationFuncs.Keys.NOTCH.value:
            value = activate_notch(raw)
        elif key == ActivationFuncs.Keys.STEP.value:
            value = activate_step(raw)

        return value


@ti.func
def activate(pop, w, inputs):
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
        act[n] = pop.activation_funcs.call(
            inputs, w, n, node.act_func, (node.gain * raw + node.bias))

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
