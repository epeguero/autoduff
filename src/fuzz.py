import torch
import torch.nn.functional as nnf
from torch.autograd.functional import jacobian, hessian
from torch.nn import Module

import tvm
import tvm.relay.transform as tvmt

from functools import reduce

import operator
import math
from bitstring import BitArray

from collections import OrderedDict

from torch_to_tvm import torch_to_tvm_mod, torch_to_tvm, torch_module_patch, eval_tvm_mod_fun, tvm_compatible_torch_funs
from autoduff_utils import generate_inputs_from_fun_sig, lookup_torch_function, lookup_torch_func
import autoduff_logging
from autoduff_logging import log

import concurrent.futures
import time
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# n == 0: identity
# n == 1: f(f(x))
# ...
def self_compose(n, f):
    def self_compose_rec(n, accum):
        if n <= 0:
            return accum
        else:
            return self_compose_rec(n-1, lambda x: f(accum(x)))
    return self_compose_rec(n, f)

def parse_torch_dtype(s):
    return {'float16':torch.float16, 'float32':torch.float32, 'float64':torch.float64}[s]

def torch_funs(f_name):
    f_torch = lookup_torch_func(f_name)
    f_grad_torch = lambda x: jacobian(f_torch, x, create_graph=True)
    f_grad2_torch= lambda x: jacobian(f_grad_torch, x)
    return f_torch, f_grad_torch, f_grad2_torch


anomaly_types = ['grad', 'fun']
loss_types = ['abs', 'squared', 'rel']
def generate_loss_funs(torch_fun_name, dtype, device, anomaly_type, loss_type):
    log.info("encoding anomaly type '{}' in loss function of type '{}'".format(anomaly_type, loss_type))

    def loss_abs(f, g, x):
        return abs(f(x) - g(x))

    def loss_abs_grad(f, f_grad, g, g_grad, x):
        return ((f(x) - g(x)) * (f_grad(x) - g_grad(x))) / (abs(f(x) - g(x)))

    def loss_squared(f, g, x):
        return (f(x) - g(x)) ** 2

    def loss_squared_grad(f, f_grad, g, g_grad, x):
        return (2 * (f(x) - g(x)) * (f_grad(x) - g_grad(x)))
        # return ((f(x) - g(x)) * (f_grad(x) - g_grad(x))) / ((f(x) - g(x)**2)**1/2)

    # def rel(a, b):
    #     return (a-b)/b
    #
    def loss_rel(f, g, x):
        return abs(f(x) - g(x)) / g(x)

    def loss_rel_grad(f, f_grad, g, g_grad, x):
        # return 2 * (f(x) - g(x)) * (g(x) * f_grad(x) - f(x) * g_grad(x)) / g(x)**3
        diff = f(x) - g(x)
        return diff * (f_grad(x) - g_grad(x)) / (g(x)**2 * abs(diff)) - (2*g_grad(x) * abs(diff)) / g_grad(x)**3

    f_torch, f_grad_torch, f_grad2_torch = torch_funs(torch_fun_name)
    f_tvm, f_grad_tvm, f_grad2_tvm = torch_to_tvm(torch_fun_name, dtype, device)

    def loss_inputs_for_anomaly_type():
        if anomaly_type == 'grad':
            return (f_grad_torch, f_grad_tvm), (f_grad_torch, f_grad2_torch, f_grad_tvm, f_grad2_tvm)
        elif anomaly_type == 'fun':
            return (f_torch, f_tvm), (f_torch, f_grad_torch, f_tvm, f_grad_tvm)
        else:
            raise Exception("Unrecognized anomaly_type: {}".format(anomaly_type))

    def loss_for_loss_type():
        if loss_type == 'abs':
            return loss_abs, loss_abs_grad
        elif loss_type == 'squared':
            return loss_squared, loss_squared_grad
        elif loss_type == 'rel':
            return loss_rel, loss_rel_grad

    loss, loss_grad = loss_for_loss_type()
    loss_inputs, loss_grad_inputs = loss_inputs_for_anomaly_type()
    return lambda x: loss(*loss_inputs, x), lambda x: loss_grad(*loss_grad_inputs, x)



def nearest_float(x, direction, dtype):
    if dtype is torch.float16:
        return nearest_half_prec_float(x, direction)

    length = int(str(dtype)[-2:])
    # print('length ', length)

    bs = BitArray(float=x.item(), length=length).bin
    # print('bs', bs)

    int_rep = int(bs, 2)
    # print('int_rep ', int_rep)

    nearest_int_rep = int_rep + (1 if direction > 0 else -1)
    # print('nearest_int_rep ', nearest_int_rep)

    nearest_bs = bin(nearest_int_rep)[2:].zfill(length)
    # print('nearest_bs ', nearest_bs)

    nearest = BitArray(bin=nearest_bs).float
    # print('x={:20}'.format(x))
    # print('nearest={:20}'.format(nearest))

    return torch.tensor(nearest)


def nearest_half_prec_float(x, direction):
    assert direction != 0, 'expected non-zero direction'

    def str_to_float16(bs):
        sign_str, e_str, m_str = bs[0], bs[1:6], bs[6:]
        sign = 1 if int(sign_str) == 0 else -1
        # accomodate subnormal numbers by checking 0 exponent string
        unbiased_e = int(e_str, 2)
        e = unbiased_e - 15 if unbiased_e != 0 else -14
        m = sum([int(m_str[i]) * 2**(-i-1) for i in range(len(m_str))]) + (1 if unbiased_e != 0 else 0)
        return sign * m * 2**e

    bs = bin(np.float16(x.item()).view('H'))[2:].zfill(16)
    nearest_int_rep = int(bs,2) + (1 if direction > 0. else -1)
    nearest_bs = bin(nearest_int_rep)[2:].zfill(16)

    return torch.tensor(str_to_float16(nearest_bs), dtype=x.dtype, device=x.device)


def fuzz_1param(loss, loss_grad, seed_input, normalized, alpha=0.4, total_iters=100, timeout=None, min_loss_perc_inc=1.):
    assert (type(seed_input) in [torch.Tensor]), "fuzzer expects seed input of Tensor type; found {}".format(type(seed_input))

    def fuzz_post_condition(fuzz_out):
        assert list(fuzz_out.values()) == sorted(list(fuzz_out.values()), key=lambda item:item[1]), "fuzzing was non-monotonic!"

    sga_step = lambda alpha, i, x, grad: (x.to(torch.float64) + alpha / (i+1) * grad.to(torch.float64)).to(original_dtype)

    original_dtype = seed_input.dtype
    original_device = seed_input.device

    x = seed_input
    fuzz = OrderedDict()

    start_time = time.perf_counter()
    for i in range(total_iters):
        if timeout and time.perf_counter() - start_time > timeout:
            break

        log.info("iteration {}/{}".format(i, total_iters))
        fuzz[i] = (x.item(), loss(x).item())

        log.info("(x, loss) = {}".format(fuzz[i]))
        grad = loss_grad(x)
        log.info("loss grad = {}".format(grad))

        if grad in [math.nan, math.inf]:
            log.error("[[ FINISHED ]]: Grad was NaN or inf")

        next_x = sga_step(alpha, i, x, grad).to(original_dtype).to(original_device)
        if normalized and next_x < 0. or next_x > 1.:
            log.info("[[ DECREASING ALPHA ]]: fuzzer running in normalized mode stepped outside of [0,1]")

        delta_x = next_x - x
        log.info("X DELTA: {:5} - {:5} = {:10g}".format(next_x, x, delta_x))


        next_loss = loss(next_x).item()
        curr_loss = fuzz[i][1]
        loss_delta = next_loss - curr_loss
        log.info("LOSS DELTA: {:3} - {:3} = {:10g}".format(next_loss, curr_loss, loss_delta))

        if next_loss > curr_loss:
            assert x != next_x, "zero change in x increased loss??!? x={}, next_x={}".format(x.item(), next_x.item())
            loss_perc_inc = (next_loss/curr_loss - 1) * 100
            log.critical("[[ FUZZING X ]]: positive loss (LOSS_DELTA={}, {:3g}% increase) with x delta = {}".format(next_loss-curr_loss, loss_perc_inc, delta_x))
            if loss_perc_inc < min_loss_perc_inc:
                log.info("[[ FINISHED ]]: did not exceed minimum loss percent change factor of {} (got:{}%)".format(min_loss_perc_inc, loss_perc_inc))
                fuzz_post_condition(fuzz)
                return fuzz
            x = next_x

        else:
            # check if gradient is non-zero but too small relative to size of x (ie, is vanishing)
            # in this case, we check the closest float in the gradient direction for loss increase
            # RATIONALE: small gradient could slowly diminish over a long interval
            # NOTE: float equality here is deliberate
            if next_x == x and grad != 0:
                near_x = nearest_float(x, grad, original_dtype)
                near_loss_delta = loss(near_x).item() - fuzz[i][1]
                if near_loss_delta > 0.:
                    log.critical("[[ FUZZING x ]]: positive loss (LOSS_DELTA={}) at nearest float".format(near_loss_delta))
                    x = near_x
                else:
                    log.info("[[ FINISHED ]]: non-positive change in loss (LOSS_DELTA={}), even at nearest float in grad direction, ({})".format(near_loss_delta, near_x))
                    fuzz_post_condition(fuzz)
                    return fuzz

            elif grad == 0:
                log.info("[[ FINISHED ]]: local maxima found")
                fuzz_post_condition(fuzz)
                return fuzz

            else: # x_delta != 0.
                log.info("[[ DECREASING ALPHA ]]: since next step would yield non-positive loss, assume step went too far.")
                alpha = alpha / 10.

    fuzz_post_condition(fuzz)
    return fuzz


def uniform_sample_float(normalized, dtype, device):
    dtype = parse_torch_dtype(dtype)
    if normalized:
        return torch.rand([], dtype=dtype, device=device)
        # exponent = (torch.rand([]) * -20).to(torch.int)
        # return torch.rand([]) * (2.** exponent)
    significand = torch.rand([], dtype=dtype, device=device)
    max_exp, exp_bias = {torch.float16: (31, 15), torch.float32: (255, 127), torch.float64: (2047, 1023)}[dtype]
    exponent = (torch.rand([], device=device) * max_exp).to(torch.int) - exp_bias
    sample = (significand * (2. ** exponent)).to(dtype).to(device)
    assert sample != math.inf, 'sampled "inf" value'
    assert sample != math.nan, 'sampled "nan" value'
    return sample


def random_search(loss, dtype, device, n=300, normalized=False, timeout=120):
    torch_dtype = parse_torch_dtype(dtype)
    loss_vals = OrderedDict()
    start_time = time.perf_counter()
    i = 0
    while time.perf_counter() - start_time < timeout:
        random_x = uniform_sample_float(normalized, dtype, device)
        loss_vals[i] = (random_x.item(), loss(random_x).item())
        i = i+1
        # log.info("(x, loss(x)) sample: {}".format(loss_vals[i]))
    log.info("max loss point found: {}".format(max(loss_vals.values(), key=lambda item: item[1])))
    return loss_vals


def random_search_large_loss_grad(dtype, device, loss_grad, normalized, timeout):
    seed = uniform_sample_float(normalized, dtype, device)
    best_sample = (seed, abs(loss_grad(seed)))
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < timeout:
        sample = uniform_sample_float(normalized, dtype, device)
        sample_grad = abs(loss_grad(sample))
        if sample_grad > best_sample[1]:
            best_sample = (sample, sample_grad)
    log.info("largest sample abs(loss gradient) found: {}".format(best_sample[1]))
    return best_sample

# To find a gradient anomaly in a model F, we will first attempt to maximize the difference in gradients between the torch and tvm implementations of F.
# This will be accomplished by performing SGA on a loss function measuring this gradient difference.
# Thus, the loss function can be given by:
# L(x) = relative_error(f(x), g(x))^2
#
# SGA steps in the direction of greatest ascent of L(x), ie, its gradient:
# SGA(time, x) = x  +  alpha * dL(x)/dx / time
def detect_grad_anomaly(torchFunName, dtype, device, anomaly_type, loss_type, mode, normalized, total_time=100, seed_input=None, seed_timeout=30, random_timeout=120):
    loss, loss_grad = generate_loss_funs(torchFunName, dtype, device, anomaly_type, loss_type)
    if 'sga' in mode:
        if not seed_input:
            log.info("randomly searching for good seed input for {} seconds (hoping for one with large loss gradient)".format(seed_timeout))
            sample, sample_loss_grad = random_search_large_loss_grad(dtype, device, loss_grad, normalized, seed_timeout)
            if sample_loss_grad.isinf().item() or sample_loss_grad.isnan().item() or sample_loss_grad.item() == 0.:
                log.info("[[ STOPPING ]]: either the provided or generated seed has bad loss gradient (0, inf, or NaN). Terminating.")
                return OrderedDict()
            seed_input = sample
        return fuzz_1param(loss, loss_grad, seed_input, normalized, total_iters=total_time, timeout=90)
    elif mode == 'random':
        return random_search(loss, dtype, device, normalized=normalized, timeout=random_timeout)
    else:
        raise Exception("unimplemented grad anomaly search mode '{}'".format(mode))


modes = ['sga', 'random']
# funs is a list of triples :: (funName, dtype, device)
def gradient_anomaly_detector(funs, anomaly_type, loss_type, mode, normalized):
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=torch.multiprocessing.get_context('spawn')) as executor:
        future_to_fun = {executor.submit(detect_grad_anomaly, *fun, anomaly_type, loss_type, mode, normalized): fun for fun in funs if fun[2] == 'cuda:0'}
        for future in concurrent.futures.as_completed(future_to_fun):
            fun = future_to_fun[future]
            try:
                search_results = future.result()
                log.info('Collected results for {}'.format(fun))
                if search_results:
                    max_point = max([(x, loss) for (x,loss) in search_results.values()], key=lambda point: point[1])
                    if max_point[1] > 0.:
                        results[fun] = max_point
                        log.warning("anomalous point in {}: {}".format(fun, max_point))

                torch.cuda.empty_cache()

                log.info('[[FINISHED]]: {}'.format(fun))
            except Exception as e:
                print("Anomaly Detector Exception:\n{}".format(str(e)))

    sorted_results = sorted(results.items(), key=lambda item: item[1][1], reverse=True)
    print(sorted_results)
    return sorted_results

def autoduff(anomaly_type, loss_type, mode, normalized, use_cached=True):
    log.info("This is Autoduff.")
    start_time = time.perf_counter()

    log.info("Autoduff: Determining functions to test...")
    funs_under_test = tvm_compatible_torch_funs(use_cached)
    log.info("Autoduff: Fuzzing '{}' functions in search for anomalies".format(len(funs_under_test)))
    anomalies = gradient_anomaly_detector(funs_under_test, anomaly_type, loss_type, mode, normalized)
    log.info("Autoduff: Writing found anomalies to file")
    write_anomalies_to_file(anomalies, anomaly_type, loss_type, mode, normalized)
    log.info("Autoduff: Done.")

    end_time = time.perf_counter()
    log.critical(f"Autoduff completed execution in {end_time - start_time:0.4f} seconds.")

def write_anomalies_to_file(results, anomaly_type, loss_type, mode, normalized):
    if not results:
        return
    results_dir = 'results'
    # count preexisting results files
    n = len([name for name in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, name))])
    results_filename = '{}_{}_{}_{}_{}.txt'.format(anomaly_type, loss_type, mode, normalized, n)
    results_path = os.path.join(results_dir, results_filename)
    log.critical("Writing Autoduff results to '{}'".format(results_path))
    with open(results_path, 'w') as f:
        lines = ['{},{}\n'.format(fun, pt) for fun, pt in results]
        f.writelines(list(map(str, results)))

def test_autoduff(anomaly_types=anomaly_types, modes=modes, normalized_modes=[True, False]):
    # disable logging for performance
    autoduff_logging.disable_most_logging(log)
    loss_type = 'abs'
    for anomaly_type in anomaly_types:
        for mode in modes:
            for normalized in normalized_modes:
                autoduff(anomaly_type, loss_type, mode, normalized, use_cached=True)




# def post_search_tuning(funName, dtype, device, x_center):
#     torch_dtype = parse_torch_dtype(dtype)
#     torch_float_apply = lambda f, x: f(x.to(torch_dtype).to(device)).item()
#     _, _, f_torch = lookup_torch_function(funName)
#     f_tvm, _, _ = torch_to_tvm(funName, dtype, device)
#
#     def torch_tvm_diff(x):
#         torch_val = torch_float_apply(f_torch, x)
#         tvm_val = torch_float_apply(f_tvm, x)
#         return torch_val - tvm_val
#
#     loss, loss_grad = generate_loss_funs(funName, dtype, device)
#
#     xs1 = [x_center * 10.**x for x in range(-50, 50)]
#     ys1 = [torch_float_apply(loss, x) for x in xs1]
#
#     max_point1 = max(zip(xs1, ys1), key=lambda point: point[1])
#
#     xs2 = [max_point1[0] * 1.5**x for x in range(-50, 50)]
#     ys2 = [torch_float_apply(loss, x) for x in xs2]
#     max_point2 = max(zip(xs2, ys2), key=lambda point: point[1])
#
#     return max_point2


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    def generate_torch_derivatives(f):
        f_grad = lambda x: jacobian(f, x, create_graph=True)
        f_grad2 = lambda x: jacobian(f_grad, x)
        return f_grad, f_grad2

    def generate_tvm_derivatives(f, dtype, device):
        _, torchFunSig, torch_fun = lookup_torch_function(f.__name__)
        x = generate_inputs_from_fun_sig(torchFunSig, dtype, device)[0] #assume arity one
        tvm_mod = torch_to_tvm_mod(torch_module_patch(f.__name__, torch_fun), x)
        tvm_mod = tvm_grad_gen(tvm_mod)

        tvm_grad = lambda x: eval_tvm_mod_fun(tvm_mod, [x], fun='grad', dtype=dtype, device=device)
        tvm_grad2 = lambda x: eval_tvm_mod_fun(tvm_mod, [x], fun='grad2', dtype=dtype, device=device)

        return tvm_grad, tvm_grad2

    def torch_derivative_test(f, f_grad, f_grad2, dtype, device):
        print("Derivative test for '{}' ({}, {})".format(f.__name__, dtype, device))

        # Code Under Test
        f_grad_test, f_grad2_test = generate_torch_derivatives(f)

        x = torch.rand((1,), dtype=getattr(torch, dtype), device=device)
        actual = f_grad_test(x)
        expected = f_grad(x)
        assert torch.isclose(actual, expected).all().item(), \
                "[[ FAIL ]]: first derivative failed; expected {}, got {}".format(expected, actual)
        print("[[ SUCCESS ]]: first pytorch derivative test passed!")

        actual = f_grad2_test(x)
        expected = f_grad2(x)
        assert torch.isclose(actual, expected).all().item(), \
                "[[ FAIL ]]: second derivative failed; expected {}, got {}".format(expected, actual)
        print("[[ SUCCESS ]]: second derivative test passed!")

    def sin_derivative_test(dtype, device):
        torch_derivative_test(torch.sin, torch.cos, lambda x: -torch.sin(x), dtype, device)

    def loss_function_test(torch_fun_name, dtype, device, compose=0):
        print("Loss function test for '{}' ({}, {})".format(torch_fun_name, dtype, device))
        _, torchFunSig, _ = lookup_torch_function(torch_fun_name)
        loss, loss_grad = generate_loss_funs(torch_fun_name, dtype, device, compose=0)
        try:
            x = generate_inputs_from_fun_sig(torchFunSig, dtype, device)[0]  #assume arity one
            loss_val = loss(x)
            loss_grad = loss_grad(x)
            print("[[ SUCCESS ]]: no exception while calculating loss and loss gradient")
            print("loss: ", loss_val)
            print("loss grad: ", loss_grad)

        except Exception as e:
            print("[[ FAIL ]]: failed with exception while calculating loss or loss gradient:")
            print(str(e))

    def sga_test(torch_fun_name, dtype, device):
        print("SGA test for '{}' ({}, {})".format(torch_fun_name, dtype, device))
        _, torchFunSig, _ = lookup_torch_function(torch_fun_name)
        loss, loss_grad = generate_loss_funs(torch_fun_name, dtype, device)
        x = generate_inputs_from_fun_sig(torchFunSig, dtype, device)[0] #assume arity one

        def sga_step(time, x, grad, alpha):
            return x + grad * alpha/(1 + time)

        try:
            new_x = sga_step(5, x, loss_grad(x), 0.01)
            print("[[ SUCCESS]]: no exception while calculating sga step")
            print("old x:", x)
            print("new x:", new_x)
        except Exception as e:
            print("[[ FAIL ]]: failed with exception while calculating sga step:")
            print(str(e))

    def detect_grad_anomaly_test(torch_fun_name, dtype, device):
        try:
            detect_grad_anomaly(torch_fun_name, dtype, device)
            print("[[ SUCCESS ]]: gradient anomaly detector finished successfully")
        except Exception as e:
            print("[[ FAIL ]]: exception while detecting gradient anomaly")
            print(str(e))

    def composed_sin_derivative_test(dtype, device):
        f = torch.sin
        composed_f = self_compose(1, f)
        grad = lambda x: torch.cos(torch.sin(x)) * torch.cos(x)
        grad2= lambda x: ((-torch.sin(torch.sin(x)) * torch.cos(x)) * torch.cos(x)) + (-torch.sin(x) * torch.cos(torch.sin(x)))
        torch_derivative_test(composed_f, grad, grad2, dtype, device)

    def tvm_vs_torch_compose_sin(n, dtype, device, f, x):
        f = self_compose(n, f)
        torch_grad, _ = generate_torch_derivatives(f)
        tvm_grad, _ = generate_tvm_derivatives(f, dtype, device)
        torch_result = torch_grad(x)
        tvm_result = tvm_grad(x)
        assert torch.isclose(torch_result, tvm_result).all().item(), "torch and tvm gradients for '{}' diverge for at composition level {}\n{}".format(f.__name__, n, torch_result-tvm_result)
        print("[[ SUCCESS ]]: torch and tvm gradients for '{}' are equal!".format(f.__name__))

    def generate_scatter_plot(ax, xs, ys, scatter_label='', title='', x_label='', y_label='', color='b', dot_size=10):
        dot_sizes = [dot_size for _ in range(len(xs))]
        ax.scatter(xs, ys, dot_sizes, linestyle='solid', color=color, label=scatter_label)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    def plot_search(ax, funName, dtype, device, title='', color='b', mode='sga'):
        search_results = detect_grad_anomaly(funName, dtype, device, mode=mode)
        xs = [x.item() for (x, _) in search_results.values()]
        ys = [y.item() for (_, y) in search_results.values()]
        generate_scatter_plot(ax, xs, ys, scatter_label=mode, color=color, title=title, x_label='x', y_label='loss(x)')

    def plot_compare_searches(funName, dtype, device):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Search Mode Comparison for '{}' ({},{})".format(funName, dtype, device))

        for color, mode in [('b', 'sga'), ('g', 'random')]:
            log.info('plotting search in mode "{}"'.format(mode))
            plot_search(ax, funName, dtype, device, color=color, mode=mode)

        plt.legend(loc='upper left')
        plt.show()

    def create_subplot(fig, dims=111, title='', xlabel='', ylabel='', log_scale=False):
        ax = fig.add_subplot(dims)
        if log_scale:
            ax.set_xscale('symlog')
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        return ax

    def plot_loss(funName, dtype, device, x_center=0):
        torch_dtype = parse_torch_dtype(dtype)
        torch_float_apply = lambda f, x: f(torch.tensor(x).to(torch_dtype).to(device)).item()
        _, _, f_torch = lookup_torch_function(funName)
        f_tvm, _, _ = torch_to_tvm(funName, dtype, device)

        def torch_tvm_diff(x):
            torch_val = torch_float_apply(f_torch, x)
            tvm_val = torch_float_apply(f_tvm, x)
            return torch_val - tvm_val

        loss, loss_grad = generate_loss_funs(funName, dtype, device)

        log.info('calculating loss values...')
        xs1 = [x_center * 10.**x for x in range(-50, 50)]
        ys1 = [torch_float_apply(loss, x) for x in xs1]

        max_point1 = max(zip(xs1, ys1), key=lambda point: point[1])
        print("largest loss at low resolution: {}".format(max_point1))

        xs2 = [max_point1[0] * 1.5**x for x in range(-50, 50)]
        ys2 = [torch_float_apply(loss, x) for x in xs2]
        max_point2 = max(zip(xs2, ys2), key=lambda point: point[1])
        print("largest loss at high resolution: {}".format(max_point2))

        print("torch_tvm max diff: {}".format(torch_tvm_diff(max_point2[0])))
        print("torch val: ({}, {})".format(max_point2[0], torch_float_apply(f_torch, max_point2[0])))
        print("tvm val: ({}, {})".format(max_point2[0], torch_float_apply(f_tvm, max_point2[0])))

        log.info('generating plots...')
        fig = plt.figure()
        ax1 = create_subplot(fig, dims=211, title="loss(x) '{}' ({},{})".format(funName, dtype, device), xlabel='x', ylabel='loss(x)', log_scale=True)
        generate_scatter_plot(ax1, xs1, ys1, color='b', dot_size=30, scatter_label='loss')

        ax2 = create_subplot(fig, dims=212, title="loss(x) '{}' ({},{})".format(funName, dtype, device), xlabel='x', ylabel='loss(x)', log_scale=True)
        generate_scatter_plot(ax2, xs2, ys2, color='b', dot_size=30, scatter_label='loss')

        # if with_grad:
        #     ys_grad = [y_grad for (_, _, y_grad) in loss_points]
        #     print("largest loss grad: {}".format(loss_points[0]))
        #     generate_scatter_plot(ax, xs, ys_grad, color='r', dot_size=10, scatter_label='loss_grad')
        # plt.xlim([min(xs), max(xs)])
        plt.show()

    def compare_torch_tvm_fun(funName, dtype, device):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('x')
        ax.set_ylabel('{}(x)'.format(funName))
        ax.set_title("{} ({}, {}) PyTorch vs TVM".format(funName, dtype, device))

        f_torch, f_grad_torch, f_grad2_torch = torch_funs(funName)
        f_tvm, f_grad_tvm, f_grad2_tvm = torch_to_tvm(funName, dtype, device)
        funs = [(f_torch, 'f_torch', 'darkred', 20),
                (f_grad_torch, 'f_grad_torch', 'red', 20),
                (f_grad2_torch, 'f_grad_torch2', 'lightcoral', 20),
                (f_tvm, 'f_tvm', 'darkgreen', 10),
                (f_grad_tvm, 'f_grad_tvm', 'forestgreen', 10),
                (f_grad2_tvm, 'f_grad2_tvm', 'limegreen', 10)]

        for f, label, color, dot_size in funs:
            xs = list(range(-100, 100, 3))
            ys = [f(torch.tensor(x).to(parse_torch_dtype(dtype)).to(device)).item() for x in xs]
            generate_scatter_plot(ax, xs, ys, scatter_label=label, color=color, dot_size=dot_size)

        plt.legend(loc='upper left')
        plt.show()


    # results :: Dict (f_name, dtype, device) (Dict Tensor Tensor)
    # The dict value (Dict Tensor Tensor) defines a search path
    def analyze_search_results(results):
        max_point = lambda points: max([point for point in points], key=lambda point: point[1])
        sorted_fun_searches = sorted([(fun, max_point(search.values())) for fun, search in results.items()], key=lambda item: item[1][1], reverse=True)
        for fun, (max_x, max_y) in sorted_fun_searches:
            if max_y > 0.:
                plot_loss(*fun, x_center=max_x)

    def eval_loss_at_point(fun_name, dtype, device, point):
        loss, _ = generate_loss_funs(fun_name, dtype, device, compose=0)
        return loss(point)

    def test_fuzzer_correctly_computes_losses(torch_fun_name, dtype, device, anomaly_type):
        f_torch, f_grad_torch, _= torch_funs(torch_fun_name)
        tvm_fun, f_grad_tvm, _= torch_to_tvm(torch_fun_name, dtype, device)
        loss, loss_grad = generate_loss_funs(torch_fun_name, dtype, device, anomaly_type)

        fuzz_out = detect_grad_anomaly(torch_fun_name, dtype, device, anomaly_type, total_time=2)
        for (x_val, actual_loss_val) in fuzz_out.values():
            x = torch.tensor(x_val)
            actual_loss = torch.tensor(actual_loss_val)
            expected_loss = loss(x)
            assert torch.isclose(actual_loss, expected_loss).item(), "incorrect loss output from fuzzer. At point {}: Expected {}; got {}".format(x, expected_loss, actual_loss)
        print("[[TEST SUCCESS ]]: fuzzer correctly computes losses")

    def test_loss_generation_consistent(torch_fun_name, dtype, device, anomaly_type):
        loss_a, loss_grad_a = generate_loss_funs(torch_fun_name, dtype, device, anomaly_type)
        loss_b, loss_grad_b = generate_loss_funs(torch_fun_name, dtype, device, anomaly_type)
        loss_c, loss_grad_c = generate_loss_funs(torch_fun_name, dtype, device, anomaly_type)
        x = torch.rand([])
        assert torch.isclose(loss_a(x), loss_b(x)) and torch.isclose(loss_b(x), loss_c(x)), "generated loss funs don't behave equivalently: {}, {}, {}".format(loss_a(x), loss_b(x), loss_c(x))
        print("[[ TEST SUCCESS ]]: generated loss functions behave equivalently at test point")


    def plot_loss(fun_name, dtype, device, anomaly_type, x_center, x_start=0., x_stop=1.):
        loss, loss_grad = generate_loss_funs(fun_name, dtype, device, anomaly_type)
        lift_float = lambda x: torch.tensor(x, dtype=parse_torch_dtype(dtype), device=device)
        xs = np.linspace(x_start, x_center, num=100) + np.linspace(x_center, x_stop, num=100)
        ys = [loss(lift_float(x)).item() for x in xs]
        plt.plot(xs, ys)
        plt.savefig('{} loss ({}, {}, {})'.format(anomaly_type, fun_name, dtype, device))

    # plot_loss('sin', 'float32', 'cuda:0', 'grad')
    # plot_loss('sin', 'float32', 'cuda:0', 'fun')


    # def test_suite():
        # test_fuzzer_correctly_computes_losses('sin', 'float32', 'cpu', normalized=True)
        # sin_derivative_test('float32', 'cpu')
        # sin_derivative_test('float32', 'cuda:0')
        # loss_function_test('sin', 'float32', 'cpu')
        # loss_function_test('sin', 'float32', 'cuda:0')
        # sga_test('sin', 'float32', 'cpu')
        # sga_test('sin', 'float32', 'cuda:0')
        # detect_grad_anomaly_test('sin', 'float32', 'cpu')
        # detect_grad_anomaly_test('sin', 'float32', 'cuda:0')
        # loss_function_test('sin', 'float32', 'cuda:0', compose=1)
        # composed_sin_derivative_test('float32', 'cpu')
        # composed_sin_derivative_test('float32', 'cuda:0')
        # tvm_vs_torch_compose_sin(0, 'float32', 'cpu', torch.sin, torch.rand((2,2)))

    test_autoduff()
    # detect_grad_anomaly('tanh', 'float16', 'cuda:0', 'grad', 'abs', mode='sga', normalized=False, seed_input=torch.tensor(95.5625, dtype=torch.float16, device='cuda:0'))
    # test_loss_generation_consistent('sin', 'float32', 'cpu', 'grad')
    # test_fuzzer_correctly_computes_losses('sin', 'float32', 'cpu', 'grad')
    # ad_plt.plot_finish(name='fuzz', save=True, show=False)
    # nearest_float(2.1118e24, 1, 'float32')

    # autoduff(use_cached=False)
    # test_suite()
    # plot_fuzz('reciprocal', 'float32', 'cpu')
    # plot_fuzz('reciprocal', 'float32', 'cuda:0')
    # plot_fuzz('sin', 'float32', 'cuda:0')
    # plot_compare_searches('tanhshrink', 'float32', 'cuda:0')
    # plot_compare_searches('reciprocal', 'float32', 'cuda:0')
    # plot_loss('reciprocal', 'float32', 'cpu')
    # plot_loss('reciprocal', 'float32', 'cuda:0')
    # plot_loss('tanhshrink', 'float32', 'cpu')
    # plot_loss('tanhshrink', 'float32', 'cuda:0')
    # gradient_anomaly_detector()
    # compare_torch_tvm_fun('tanhshrink', 'float32', 'cpu')
    # plot_loss('reciprocal', 'float32', 'cpu')
    # compare_torch_tvm_fun('reciprocal', 'float32', 'cpu')
    # compare_torch_tvm_fun('tanhshrink', 'float32', 'cuda:0')
    # results = gradient_anomaly_detector_par()
    # analyze_search_results({('tan', 'float32', 'cpu') : {0 : (torch.tensor(-1.3110e15), torch.tensor(3.9063e11))}})
