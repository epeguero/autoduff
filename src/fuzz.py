import torch
import torch.nn.functional as nnf
from torch.autograd.functional import jacobian, hessian
from torch.nn import Module

import tvm
import tvm.relay.transform as tvmt

from functools import reduce

import operator
import math

from collections import OrderedDict

from torch_to_tvm import torch_to_tvm_mod, torch_to_tvm, torch_module_patch, eval_tvm_mod_fun, tvm_grad_gen, tvm_compatible_torch_funs
from autoduff_utils import generate_inputs_from_fun_sig, lookup_torch_function, lookup_torch_func
from autoduff_logging import log

import concurrent.futures
import time
import os

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

import bitstring
binary = lambda x: bitstring.BitArray(float=x, length=32).bin

LOSS_SCALING = 1e14
def generate_loss_funs(torch_fun_name, dtype, device, compose=0):

    def to_sum_of_entries(f):
        return lambda x: f(x).sum()

    def loss(f, g, x):
        return abs(f(x) - g(x)) * LOSS_SCALING

    def loss_grad(f, f_grad, g, g_grad, x):
        return (LOSS_SCALING * ((f(x) - g(x)) * (f_grad(x) - g_grad(x)))) / (abs(f(x) - g(x)) * LOSS_SCALING) * LOSS_SCALING

    # def rel(a, b):
    #     return (a-b)/b
    #
    # def loss(f, g, x):
    #     return scaling * rel(f(x), g(x)) ** 2
    #
    # def loss_grad(f, f_grad, g, g_grad, x):
    #     return scaling * 2 * (f(x) - g(x)) * (g(x) * f_grad(x) - f(x) * g_grad(x)) / g(x)**3


    _, torchFunSig, torch_fun = lookup_torch_function(torch_fun_name)
    torch_fun = self_compose(compose, torch_fun)
    x = generate_inputs_from_fun_sig(torchFunSig, dtype, device)[0] #assume arity one
    tvm_mod = torch_to_tvm_mod(torch_module_patch(torch_fun_name, torch_fun), x)
    tvm_mod = tvm_grad_gen(tvm_mod)

    torch_grad = lambda x: jacobian(torch_fun, x, create_graph=True)
    torch_grad2 = lambda x: jacobian(torch_grad, x)
    tvm_grad = lambda x: eval_tvm_mod_fun(tvm_mod, [x], fun='grad', dtype=dtype, device=device)
    tvm_grad2 = lambda x: eval_tvm_mod_fun(tvm_mod, [x], fun='grad2', dtype=dtype, device=device)

    torch_grad, torch_grad2, tvm_grad, tvm_grad2 = \
        list(map(to_sum_of_entries, [torch_grad, torch_grad2, tvm_grad, tvm_grad2]))

    return (lambda x: loss(torch_grad, tvm_grad, x),
            lambda x: loss_grad(torch_grad, torch_grad2, tvm_grad, tvm_grad2, x))


# loss :: Tensor -> Tensor
# seed :: tuple Tensor   NOTE: Must be Tuple!
def fuzz(loss, sgx_step, seed_inputs, loss_grad_fun = None, total_time=100):
    assert (type(seed_inputs) in [tuple, list]), "fuzzer expects tuple type of seed inputs; found {}".format(type(seed_inputs))
    loss_grads = (lambda xs: jacobian(loss, tuple(xs))) if loss_grad_fun is None else loss_grad_fun
    original_dtypes = [x.dtype for x in seed_inputs]
    downcast = lambda xs: [x.to(dtype) for x, dtype in zip(xs, original_dtypes)]

    xs = seed_inputs
    loss_vals = OrderedDict()
    for time in range(total_time):
        log.info("iteration {}/{}".format(time, total_time))
        loss_vals[time] = loss(xs)

        log.info("loss = {}".format(loss_vals[time]))
        with torch.no_grad():
            print(xs)
            next_xs = [sgx_step(time, x.to(torch.float64), grad.to(torch.float64)) for x, grad in zip(xs, loss_grads(xs))]
            xs = downcast(next_xs)

        if any(map(lambda t:torch.isnan(t).any(), xs)):
            raise Exception("fuzzer encountered NaN value at iteration {}".format(time))

    return xs, loss_vals


def fuzz_1param(loss, sgx_step, seed_input, loss_grad_fun, total_time=100, normalized=False):
    assert (type(seed_input) in [torch.Tensor]), "1 param fuzzer expects seed input of Tensor type; found {}".format(type(seed_input))
    original_dtype = seed_input.dtype
    downcast = lambda x: x.to(original_dtype)

    x = seed_input
    loss_vals = OrderedDict()
    for time in range(total_time):
        # log.info("iteration {}/{}".format(time, total_time))
        loss_vals[time] = (x.item(), loss(x).item())

        # log.info("x = {}".format(x))
        # log.info("loss = {}".format(loss_vals[time][1]))
        with torch.no_grad():
            # log.info('gradient: {}'.format(binary(np.float32(loss_grad_fun(x).item()))))
            # log.info('gradient: {}'.format(loss_grad_fun(x).item()))
            next_x = sgx_step(time, x.to(torch.float64), loss_grad_fun(x).to(torch.float64))
            x = downcast(next_x)

            if torch.isnan(x).sum() > 0:
                log.info("[[ NaN ]]: Ending fuzz at iteration {}".format(time))
                return loss_vals

            # end normalized sga search if we escape [0, 1)
            # We could continue with a random seed, but then fuzzing can devolve into random search
            # This is not necessarily bad, but we avoid that to distinguish performance of sga from random search
            if normalized and (x < 0. or x >= 1.):
                # x = torch.rand(x.shape, dtype=x.dtype, device=x.device)
                return loss_vals

    return loss_vals


def random_search(loss, dtype, device, n=100, normalized=True):
    torch_dtype = parse_torch_dtype(dtype)
    min_x, max_x = 0., 1. if normalized else -100., 100.
    loss_vals = OrderedDict()
    for i in range(n):
        random_x = (max_x - min_x) * torch.rand([]).to(torch_dtype).to(device) + min_x
        loss_vals[i] = (random_x, loss(random_x))
    return loss_vals


# To find a gradient anomaly in a model F, we will first attempt to maximize the difference in gradients between the torch and tvm implementations of F.
# This will be accomplished by performing SGA on a loss function measuring this gradient difference.
# Thus, the loss function can be given by:
# L(x) = relative_error(f(x), g(x))^2
#
# SGA steps in the direction of greatest ascent of L(x), ie, its gradient:
# SGA(time, x) = x  +  alpha * dL(x)/dx / time
def detect_grad_anomaly(torchFunName, dtype, device, mode='sga', normalized=False):
    if 'sga' in mode:
        loss_fun, loss_grad_fun = generate_loss_funs(torchFunName, dtype, device)
        alpha = 1.2
        sga_step = lambda time, x, grad: x + grad
        _, torchFunSig, torch_fun = lookup_torch_function(torchFunName)
        seed_inputs = generate_inputs_from_fun_sig(torchFunSig, dtype, device)
        return fuzz_1param(loss_fun, sga_step, seed_inputs[0], loss_grad_fun, total_time=100)
    elif mode == 'random':
        return random_search(loss_fun, dtype, device)
    else:
        raise Exception("unimplemented grad anomaly search mode '{}'".format(mode))


def gradient_anomaly_detector():
    results = []
    for (f, dtype, device) in tvm_compatible_torch_funs():
        try:
            loss_vals = detect_grad_anomaly(f, dtype, device)
            worst_loss = max([(_, loss_val) for (_, loss_val) in loss_vals.values()])
            results.append(((f, dtype, device), worst_loss))
        except Exception as e:
            print("Anomaly Detector Exception:\n{}".format(str(e)))
    results.sort(key = lambda x: x[1], reverse=True)
    print("\n".join(list(map(lambda x: str(x), results))))
    return results



def gradient_anomaly_detector_par(mode='sga', normalized=False, strict=False):
    results = {}
    # executor = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=torch.multiprocessing.get_context('spawn'))
    # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # search_algo = lambda *fun: detect_grad_anomaly(*fun, mode=mode)
        # future_to_fun = {executor.submit(search_algo, *fun): fun for fun in tvm_compatible_torch_funs()}
        # TODO: get mode keyword to work!
    with concurrent.futures.ProcessPoolExecutor(max_workers=12, mp_context=torch.multiprocessing.get_context('spawn')) as executor:
        future_to_fun = {executor.submit(detect_grad_anomaly, *fun, mode=mode): fun for fun in tvm_compatible_torch_funs() if fun[2] == 'cuda:0'}
        for future in concurrent.futures.as_completed(future_to_fun):
            fun = future_to_fun[future]
            try:
                search_results = future.result()
                log.info('Collected results for {}'.format(fun))
                max_point = max([(x, loss/LOSS_SCALING) for (x,loss) in search_results.values()], key=lambda point: point[1])
                if max_point[1] > 0.:
                    results[fun] = max_point
                    log.warning("anomalous point in {}: {}".format(fun, max_point))

                # log.info('Tuning fuzzed point for {}'.format(fun))
                # tuned_max_point = post_search_tuning(*fun, max_point[0])
                # if tuned_max_point[1] > 0.:
                #     results[fun] = tuned_max_point.to('cpu')
                #     log.info("max point: ", tuned_max_point)
                torch.cuda.empty_cache()
                # TODO: implement strict sga. The priority here is to strictly use sga to find a maximum (as a means of measuring sga's effectiveness),
                # rather than to find the maximum among all seen points
                # if strict and mode=='sga_strict':
                #     last_point = list(search_results.values(0))[-1]

                log.info('[[FINISHED]]: {}'.format(fun))
            except Exception as e:
                print("Anomaly Detector Exception:\n{}".format(str(e)))
                torch.cuda.empty()
                # TODO: implement strict sga. The priority here is to strictly use sga to find a maximum (as a means of measuring sga's effectiveness),
                raise

    sorted_results = sorted(results.items(), key=lambda item: item[1][1], reverse=True)
    print(sorted_results)
    return sorted_results

def autoduff():
    log.info("This is Autoduff.")
    start_time = time.perf_counter()
    write_results_to_file(gradient_anomaly_detector_par(mode='sga'))
    end_time = time.perf_counter()
    log.info(f"Autoduff completed execution in {end_time - start_time:0.4f} seconds.")


def write_results_to_file(results):
    results_dir = 'results'
    # count preexisting results files
    n = len([name for name in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, name))])
    results_filename = 'autoduff_results{}.csv'.format(n)
    results_path = os.path.join(results_dir, results_filename)
    log.info("Writing Autoduff results to '{}'".format(results_path))
    with open(results_path, 'w') as f:
        lines = ['{},{}\n'.format(fun, pt) for fun, pt in results]
        f.writelines(list(map(str, results)))


def post_search_tuning(funName, dtype, device, x_center):
    torch_dtype = parse_torch_dtype(dtype)
    torch_float_apply = lambda f, x: f(x.to(torch_dtype).to(device)).item()
    _, _, f_torch = lookup_torch_function(funName)
    f_tvm, _, _ = torch_to_tvm(funName, dtype, device)

    def torch_tvm_diff(x):
        torch_val = torch_float_apply(f_torch, x)
        tvm_val = torch_float_apply(f_tvm, x)
        return torch_val - tvm_val

    loss, loss_grad = generate_loss_funs(funName, dtype, device)

    xs1 = [x_center * 10.**x for x in range(-50, 50)]
    ys1 = [torch_float_apply(loss, x) for x in xs1]

    max_point1 = max(zip(xs1, ys1), key=lambda point: point[1])

    xs2 = [max_point1[0] * 1.5**x for x in range(-50, 50)]
    ys2 = [torch_float_apply(loss, x) for x in xs2]
    max_point2 = max(zip(xs2, ys2), key=lambda point: point[1])

    return max_point2


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
        return loss(point) / LOSS_SCALING


    def test_suite():
        sin_derivative_test('float32', 'cpu')
        sin_derivative_test('float32', 'cuda:0')
        loss_function_test('sin', 'float32', 'cpu')
        loss_function_test('sin', 'float32', 'cuda:0')
        sga_test('sin', 'float32', 'cpu')
        sga_test('sin', 'float32', 'cuda:0')
        detect_grad_anomaly_test('sin', 'float32', 'cpu')
        detect_grad_anomaly_test('sin', 'float32', 'cuda:0')
        loss_function_test('sin', 'float32', 'cuda:0', compose=1)
        composed_sin_derivative_test('float32', 'cpu')
        composed_sin_derivative_test('float32', 'cuda:0')
        # tvm_vs_torch_compose_sin(0, 'float32', 'cpu', torch.sin, torch.rand((2,2)))

    autoduff()
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
