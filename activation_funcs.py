from enum import Enum

import taichi as ti

# TODO: Are these the right activations? All taken from neat-python package.
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


@ti.func
def call(act_func, raw):
    result = 0.0
    if act_func == 0:
        result = activate_sigmoid(raw)
    elif act_func == 1:
        result = activate_tanh(raw)
    elif act_func == 2:
        result = activate_sin(raw)
    elif act_func == 3:
        result = activate_gaus(raw)
    elif act_func == 4:
        result = activate_relu(raw)
    elif act_func == 5:
        result = activate_elu(raw)
    elif act_func == 6:
        result = activate_lelu(raw)
    elif act_func == 7:
        result = activate_selu(raw)
    elif act_func == 8:
        result = activate_softplus(raw)
    elif act_func == 9:
        result = activate_identity(raw)
    elif act_func == 10:
        result = activate_clamped(raw)
    elif act_func == 11:
        result = activate_inv(raw)
    elif act_func == 12:
        result = activate_log(raw)
    elif act_func == 13:
        result = activate_exp(raw)
    elif act_func == 14:
        result = activate_abs(raw)
    elif act_func == 15:
        result = activate_hat(raw)
    elif act_func == 16:
        result = activate_square(raw)
    elif act_func == 17:
        result = activate_cube(raw)
    return result
