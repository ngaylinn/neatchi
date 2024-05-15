"""Activation functions for the CPPNs.

In a CPPN, each neuron has its own activation function, chosen from the
ActivationFuncs enum. Use the random() function to get a randomly chosen
activation function and the call() function to invoke one by its enum value.
"""

from enum import Enum

import taichi as ti

# TODO: Are these the right activations? All taken from neat-python package.
# TODO: Where do all the inline constants come from? Are they appropriate?
# TODO: Consider letting the caller choose a set of activation functions.
class ActivationFuncs(Enum):
    SIGMOID = 0
    TANH = 1
    SIN = 2
    GAUS = 3
    RELU = 4
    ELU = 5
    LELU = 6
    SELU = 7
    SOFTPLUS = 8
    IDENTITY = 9
    CLAMPED = 10
    INV = 11
    LOG = 12
    EXP = 13
    ABS = 14
    HAT = 15
    SQUARE = 16
    CUBE = 17

NUM_ACTIVATION_FUNCS = len(ActivationFuncs.__members__)

@ti.func
def random():
    return ti.random(dtype=int) % ti.static(len(ActivationFuncs))

@ti.func
def abs_clamp(value, extreme):
    return ti.max(-extreme, ti.min(extreme, value))

@ti.func
def activate_sigmoid(raw):
    return 1.0 / (1.0 + ti.exp(-abs_clamp(5.0 * raw, 60.0)))

@ti.func
def activate_tanh(raw):
    return ti.tanh(abs_clamp(5.0 * raw, 60.0))

@ti.func
def activate_sin(raw):
    return ti.sin(abs_clamp(5.0 * raw, 60.0))

@ti.func
def activate_gaus(raw):
    return ti.exp(-5 * abs_clamp(raw, 3.4)) ** 2

@ti.func
def activate_relu(raw):
    return (raw > 0.0) * raw

@ti.func
def activate_elu(raw):
    return ti.select(raw > 0.0, raw, ti.exp(raw) - 1)

@ti.func
def activate_lelu(raw):
    return ti.select(raw > 0.0, raw, 0.005 * raw)

@ti.func
def activate_selu(raw):
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return ti.select(raw > 0.0, lam * raw, lam * alpha * (ti.exp(raw) - 1))

@ti.func
def activate_softplus(raw):
    return 0.2 * ti.log(1 + ti.exp(abs_clamp(5.0 * raw, 60.0)))

@ti.func
def activate_identity(raw):
    return raw

@ti.func
def activate_clamped(raw):
    return abs_clamp(raw, 1.0)

@ti.func
def activate_inv(raw):
    return ti.select(raw != 0, 1.0 / raw, raw)

@ti.func
def activate_log(raw):
    return ti.log(max(1e-7, raw))

@ti.func
def activate_exp(raw):
    return ti.exp(abs_clamp(raw, 60.0))

@ti.func
def activate_abs(raw):
    return ti.abs(raw)

@ti.func
def activate_hat(raw):
    return ti.max(0.0, 1 - ti.abs(raw))

@ti.func
def activate_square(raw):
    return raw ** 2

@ti.func
def activate_cube(raw):
    return raw ** 3


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
    if act_func == ActivationFuncs.ELU.value:
        return activate_elu(raw)
    if act_func == ActivationFuncs.LELU.value:
        return activate_lelu(raw)
    if act_func == ActivationFuncs.SELU.value:
        return activate_selu(raw)
    if act_func == ActivationFuncs.SOFTPLUS.value:
        return activate_softplus(raw)
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
    if act_func == ActivationFuncs.HAT.value:
        return activate_hat(raw)
    if act_func == ActivationFuncs.SQUARE.value:
        return activate_square(raw)
    if act_func == ActivationFuncs.CUBE.value:
        return activate_cube(raw)
