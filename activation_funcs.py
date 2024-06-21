"""Activation functions for the CPPNs.

In a CPPN, each neuron has its own activation function, chosen from the
ActivationFuncs enum. Use the random() function to get a randomly chosen
activation function and the call() function to invoke one by its enum value.
"""

from enum import Enum

import taichi as ti


# These functions have been tuned arbitrarily to have an "appropriate" range of
# values when applied to values in the range from 0.0 to 1.0.
# TODO: Consider letting the caller choose a set of activation functions.
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
    SQUARE_WAVE = 15
    NOTCH = 16
    STEP = 17

NUM_ACTIVATION_FUNCS = len(ActivationFuncs.__members__)


@ti.func
def random():
    return ti.random(dtype=int) % ti.static(NUM_ACTIVATION_FUNCS)

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
def activate_square_wave(raw):
    return 2 * ((raw % 1) > 0.5) - 1

@ti.func
def activate_notch(raw):
    return raw > -0.1 and raw < 0.1

@ti.func
def activate_step(raw):
    return raw // 1.0


# TODO: Should this be a regular @ti.func?
# TODO: Should you use ti.static on the ifs?
@ti.real_func
def call(act_func: int, raw: float) -> float:
    if act_func == ActivationFuncs.SIGMOID.value:
        return activate_sigmoid(raw)
    if act_func == ActivationFuncs.TANH.value:
        return activate_tanh(raw)
    if act_func == ActivationFuncs.SIN.value:
        return activate_sin(raw)
    if act_func == ActivationFuncs.GAUS.value:
        return activate_gaus(raw)
    if act_func == ActivationFuncs.RELU.value:
        return activate_relu(raw)
    if act_func == ActivationFuncs.IDENTITY.value:
        return activate_identity(raw)
    if act_func == ActivationFuncs.CLAMPED.value:
        return activate_clamped(raw)
    if act_func == ActivationFuncs.INV.value:
        return activate_inv(raw)
    if act_func == ActivationFuncs.LOG.value:
        return activate_log(raw)
    if act_func == ActivationFuncs.EXP.value:
        return activate_exp(raw)
    if act_func == ActivationFuncs.ABS.value:
        return activate_abs(raw)
    if act_func == ActivationFuncs.SQUARE.value:
        return activate_square(raw)
    if act_func == ActivationFuncs.CUBE.value:
        return activate_cube(raw)
    if act_func == ActivationFuncs.TRIANGLE.value:
        return activate_triangle(raw)
    if act_func == ActivationFuncs.SAWTOOTH.value:
        return activate_sawtooth(raw)
    if act_func == ActivationFuncs.SQUARE_WAVE.value:
        return activate_square_wave(raw)
    if act_func == ActivationFuncs.NOTCH.value:
        return activate_notch(raw)
    if act_func == ActivationFuncs.STEP.value:
        return activate_step(raw)


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
