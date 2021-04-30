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

import pickle
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
        return ((f(x) - g(x)) / g(x)) ** 2
        # return abs(f(x) - g(x)) / g(x)

    def loss_rel_grad(f, f_grad, g, g_grad, x):
        return 2 * (f(x) - g(x)) * (g(x) * f_grad(x) - f(x) * g_grad(x)) / (g(x) ** 3)
        # return 2 * (f(x) - g(x)) * (g(x) * f_grad(x) - f(x) * g_grad(x)) / g(x) / g(x) / g(x)
        # diff = f(x) - g(x)
        # return diff * (f_grad(x) - g_grad(x)) / (g(x)**2 * abs(diff)) - (2*g_grad(x) * abs(diff)) / g_grad(x)**3

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
            break

        next_x = sga_step(alpha, i, x, grad).to(original_dtype).to(original_device)
        if normalized and next_x < 0. or next_x > 1.:
            log.info("[[ DECREASING ALPHA ]]: fuzzer running in normalized mode stepped outside of [0,1]")
            alpha = alpha / 10.
            continue

        delta_x = next_x - x
        log.info("X DELTA: {:5} - {:5} = {:10g}".format(next_x, x, delta_x))


        next_loss = loss(next_x).item()
        curr_loss = fuzz[i][1]
        loss_delta = next_loss - curr_loss
        log.info("LOSS DELTA: {:3} - {:3} = {:10g}".format(next_loss, curr_loss, loss_delta))

        if next_loss in [math.nan, math.inf]:
            log.info("[[ DECREASING ALPHA ]]: since next step would yield NaN or inf")
            alpha = alpha / 10.

        elif next_loss > curr_loss:
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
    # max_exp, exp_bias = {torch.float16: (31, 15), torch.float32: (255, 127), torch.float64: (2047, 1023)}[dtype]
    min_exp, max_exp = {torch.float16: (-14, 15), torch.float32: (-126, 127), torch.float64: (-126, 127)}[dtype]#(-16382, 16383)}[dtype]
    exponent = ((torch.rand([], device=device) * (max_exp-min_exp)).to(torch.int) + min_exp).to(dtype)
    significand = torch.rand([], dtype=dtype, device=device) + torch.tensor(0. if exponent == max_exp else 1., dtype=dtype, device=device)
    sample = (significand * (2. ** exponent)).to(dtype).to(device)
    assert sample != math.inf, 'sampled "inf" value'
    assert sample != math.nan, 'sampled "nan" value'
    return sample


def random_search(loss, dtype, device, normalized, timeout):
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
def detect_grad_anomaly(torchFunName, dtype, device, anomaly_type, loss_type, mode, normalized, total_time=100, seed_input=None, sga_timeout=60, random_timeout=60):
    log.info(f'Detecting anomalies in ({torchFunName}, {dtype}, {device})')
    log.info(f'Anomaly detection parameters: anomaly_type={anomaly_type}, mode={mode}, normalized={normalized}')
    loss, loss_grad = generate_loss_funs(torchFunName, dtype, device, anomaly_type, loss_type)
    if 'sga' in mode:
        if not seed_input:
            seed_timeout = sga_timeout / 2.
            log.info("randomly searching for good seed input for {} seconds (hoping for one with large loss gradient)".format(seed_timeout))
            sample, sample_loss_grad = random_search_large_loss_grad(dtype, device, loss_grad, normalized, seed_timeout)
            if sample_loss_grad.isinf().item() or sample_loss_grad.isnan().item() or sample_loss_grad.item() == 0.:
                log.info("[[ STOPPING ]]: either the provided or generated seed has bad loss gradient (0, inf, or NaN). Terminating.")
                return OrderedDict()
            seed_input = sample
        return fuzz_1param(loss, loss_grad, seed_input, normalized, total_iters=total_time, timeout=sga_timeout/2)
    elif mode == 'random':
        return random_search(loss, dtype, device, normalized, random_timeout)
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

results_dir = 'results'
def write_anomalies_to_file(results, anomaly_type, loss_type, mode, normalized):
    if not results:
        return
    # count preexisting results files
    n = len([name for name in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, name))])
    results_filename = '{}_{}_{}_{}_{}.txt'.format(anomaly_type, loss_type, mode, normalized, n)
    results_path = os.path.join(results_dir, results_filename)
    log.critical("Writing Autoduff results to '{}'".format(results_path))
    with open(results_path, 'wb') as f:
        pickle.dump( results, f)
        # lines = ['{},{}\n'.format(fun, pt) for fun, pt in results]
        # f.writelines(list(map(str, results)))

def read_results_from_file_with_suffix(n):
    def read_number_in_string(string):
        return int(''.join([c for c in string if c.isdigit()]))
    matching_files = [filename for filename in os.listdir(results_dir) if filename != '' and read_number_in_string(filename) == n]
    assert len(matching_files) == 1, 'multiple matches; results would be ambiguous'
    filename = matching_files[0]
    filename_minus_ext = filename.split('.')[0]
    anomaly_type, _, mode, normalize_str, n_str = filename_minus_ext.split('_')
    assert n == int(n_str), f'expected {n} at the end of the filename'
    print(f'unpickling results in file "{filename}"')
    return anomaly_type, mode, bool(normalize_str), pickle.load( open(os.path.join(results_dir, filename), 'rb') )

def analyze_results(n):
    anomaly_type, _, _, results = read_results_from_file_with_suffix(n)
    results = list(filter(lambda item: item[1][1] not in [math.nan, math.inf], results))
    extended_results = []
    for (fun, anom) in results:
        _, dtype, device = fun
        abs_err, _ = generate_loss_funs(*fun, anomaly_type, 'abs')
        abs_err_val = abs_err(torch.tensor(anom[0], dtype=parse_torch_dtype(dtype), device=device)).item()
        extended_results.append((fun, anom, abs_err_val))

    num_anomalies = len(results)

    print(f'analyzing file with #{n}:...')
    print(f'# of anomalies found: {num_anomalies}')

    (largest_abs_fun, (largest_abs_x, largest_abs_rel), largest_abs) = max(extended_results, key=lambda item: item[2])
    print(f'largest anomaly absolute error found: fun = {largest_abs_fun}; {largest_abs} @ {largest_abs_x}; rel_err = {math.sqrt(largest_abs_rel)}')

    (smallest_abs_fun, (smallest_abs_x, smallest_abs_rel), smallest_abs) = min(extended_results, key=lambda item: item[2])
    print(f'smallest anomaly absolute error found: fun = {smallest_abs_fun}; {smallest_abs} @ {smallest_abs_x}; rel_err = {math.sqrt(smallest_abs_rel)}')

    print(f'average anomaly absolute error found: {sum([item[2] for item in extended_results]) / num_anomalies}')

    (largest_rel_fun, (largest_rel_x, largest_rel), largest_rel_abs) = max(extended_results, key=lambda item: item[1][1])
    print(f'largest anomaly relative error found: fun = {largest_rel_fun}; {math.sqrt(largest_rel)} @ {largest_rel_x}; abs_err = {largest_rel_abs}')

    (smallest_rel_fun, (smallest_rel_x, smallest_rel), smallest_rel_abs) = min(extended_results, key=lambda item: item[1][1])
    print(f'smallest anomaly relative error found: fun = {smallest_rel_fun}; {math.sqrt(smallest_rel)} @ {smallest_rel_x}; abs_err = {smallest_rel_abs}')

    print(f'average anomaly relative error found: {sum([item[1][1] for item in extended_results]) / num_anomalies}')


def compare_results(n1, n2):
    _, res1 = read_file_from_number_suffix(n1)
    _, res2 = read_file_from_number_suffix(n1)
    res1_dict, res2_dict = dict(res1), dict(res2)
    score1, score2 = 0, 0
    ties = 0
    for fun in res1_dict.keys():
        if fun in res2_dict.keys():
            anom1, anom2 = res1_dicts[fun], res2_dict[fun]
            if anom1 > anom2:
                score1 += 1
            elif anom1 < anom2:
                score2 += 1
            else:
                ties += 1
    print(f'file #{n1} score: {score1}')
    print(f'file #{n2} score: {score2}')
    print(f'ties: {ties}')


def test_autoduff(anomaly_types=anomaly_types, modes=modes, normalized_modes=[True, False], loss_type = 'rel', use_cached=True):
    # disable logging for performance
    autoduff_logging.disable_most_logging(log)
    for anomaly_type in anomaly_types:
        for mode in modes:
            for normalized in normalized_modes:
                autoduff(anomaly_type, loss_type, mode, normalized, use_cached)




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


    def test_loss_fun_does_compute_loss(torch_fun_name, dtype, device, anomaly_type):
        loss, loss_grad = generate_loss_funs(torch_fun_name, dtype, device, anomaly_type, 'abs')
        f_torch, f_grad_torch, _= torch_funs(torch_fun_name)
        f_tvm, f_grad_tvm, _= torch_to_tvm(torch_fun_name, dtype, device)
        x = torch.rand([], dtype=parse_torch_dtype(dtype), device=device)
        actual = loss(x)
        expected = abs(f_torch(x) - f_tvm(x)) if anomaly_type == 'fun' else abs(f_grad_torch(x) - f_grad_tvm(x))
        assert torch.isclose(actual, expected), f"loss fun didn't compute expected value: actual={actual}, expected={expected}"
        log.critical(f"[[ TEST SUCCESS ]]: loss function with anomaly type {anomaly_type} works as expected")


    def test_fuzzer_correctly_computes_losses(torch_fun_name, dtype, device, anomaly_type, mode):
        log.info("testing that fuzzer output loss values match loss function output")
        loss, loss_grad = generate_loss_funs(torch_fun_name, dtype, device, anomaly_type, 'abs')

        def test(fuzz_out):
            for (x_val, actual_loss_val) in fuzz_out.values():
                x = torch.tensor(x_val, dtype=parse_torch_dtype(dtype), device=device)
                actual_loss = torch.tensor(actual_loss_val, dtype=parse_torch_dtype(dtype), device=device)
                expected_loss = loss(x)
                assert torch.isclose(actual_loss, expected_loss).item(), "incorrect loss output from fuzzer. At point {}: Expected {}; got {}".format(x, expected_loss, actual_loss)

        fuzz_out = detect_grad_anomaly(torch_fun_name, dtype, device, anomaly_type, 'abs', mode, False, total_time=2, sga_timeout=10, random_timeout=10)
        test(fuzz_out)

        log.critical("[[TEST SUCCESS ]]: fuzzer correctly computes losses")

    def test_loss_generation_consistent(torch_fun_name, dtype, device, anomaly_type):
        log.info("testing that multiple generated versions of loss funs with same parameters behave identically at random point")
        log.info(f"parameters: ({torch_fun_name}, {dtype}, {device}), anomaly_type={anomaly_type}")
        loss_a, loss_grad_a = generate_loss_funs(torch_fun_name, dtype, device, anomaly_type, 'abs')
        loss_b, loss_grad_b = generate_loss_funs(torch_fun_name, dtype, device, anomaly_type, 'abs')
        loss_c, loss_grad_c = generate_loss_funs(torch_fun_name, dtype, device, anomaly_type, 'abs')
        x = torch.rand([], dtype=parse_torch_dtype(dtype), device=device)
        assert torch.isclose(loss_a(x), loss_b(x), equal_nan=True) and torch.isclose(loss_b(x), loss_c(x), equal_nan=True), "generated loss funs don't behave equivalently: {}, {}, {}".format(loss_a(x), loss_b(x), loss_c(x))
        assert torch.isclose(loss_grad_a(x), loss_grad_b(x), equal_nan=True) and torch.isclose(loss_grad_b(x), loss_grad_c(x), equal_nan=True), "generated loss funs don't behave equivalently: {}, {}, {}".format(loss_grad_a(x), loss_grad_b(x), loss_grad_c(x))
        log.critical("[[ TEST SUCCESS ]]: generated loss functions behave equivalently at test point")


    def plot_loss(fun_name, dtype, device, x_center, win_size=100., n=20):
        print(f'plotting losses for ({fun_name}, {dtype}, {device})')
        # f_torch, f_grad_torch, _ = torch_funs(fun_name)
        f_tvm, f_grad_tvm, _ = torch_to_tvm(fun_name, dtype, device)
        # loss_rel, loss_rel_grad = generate_loss_funs(fun_name, dtype, device, 'grad', 'rel')
        lift_float = lambda x: torch.tensor(x, dtype=parse_torch_dtype(dtype), device=device)

        xs = np.linspace(x_center-win_size/2, x_center, num=n//2) + np.linspace(x_center, x_center+win_size/2, num=n//2)

        print(f'calculating loss + torch vals + tvm vals')
        # loss_rel_ys = [loss_rel(lift_float(x)).item() for x in xs]
        # loss_rel_grad_ys = [loss_rel_grad(lift_float(x)).item() for x in xs]
        # f_torch_ys = [f_torch(lift_float(x)).item() for x in xs]
        # f_tvm_ys = [f_tvm(lift_float(x)).item() for x in xs]
        # f_grad_torch_ys = [f_grad_torch(lift_float(x)).item() for x in xs]
        f_grad_tvm_ys = [f_grad_tvm(lift_float(x)).item() for x in xs]

        print('plotting calculated values')
        # plt.plot(xs, loss_rel_ys, color='k')
        # plt.plot(xs, loss_rel_grad_ys, color='c')
        # plt.plot(xs, f_torch_ys, color='r')
        # plt.plot(xs, f_tvm_ys, color='g')
        # plt.plot(xs, f_grad_torch_ys, color='m')
        plt.plot(xs, f_grad_tvm_ys, color='y')
        plt.savefig('{}_{}_{}'.format(fun_name, dtype, device))

    # test_loss_generation_consistent('reciprocal', 'float64', 'cuda:0', 'fun')
    # test_loss_generation_consistent('reciprocal', 'float64', 'cuda:0', 'grad')
    # test_fuzzer_correctly_computes_losses('reciprocal', 'float64', 'cuda:0', 'grad', 'sga')
    # plot_loss('reciprocal', 'float64', 'cuda:0', 0.)

    test_autoduff(use_cached=False)

