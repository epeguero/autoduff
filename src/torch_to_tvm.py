import torch
import torch.nn
import torch.nn.functional as nnf
from torch.autograd.functional import jacobian

import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine
from tvm.relay.testing import run_infer_type
from autoduff_utils import convert_all_to_single_arity, lookup_torch_func, lookup_torch_fun_sig, check_property_funs, generate_inputs_from_fun_sig, lookup_torch_function
from autoduff_logging import log, enable_logger_debug

from tvm import transform

import math
import os.path
from os import path
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# PATCH:
# some imported functions have a missing '__module__',
# which is expected by the torch to tvm conversion
def torch_module_patch(modelName, model):
    import torch
    import torch.nn
    import torch.nn.functional
    if model.__module__:
        return model
    elif hasattr(torch.nn.functional, modelName):
        model.__module__ = "torch.nn.functional"
    elif hasattr(torch.nn, modelName):
        model.__module__ = "torch.nn"
    elif hasattr(torch, modelName):
        model.__module__ = "torch"
    return model


def torch_to_tvm(torch_fun_name, dtype, device):
    _, torchFunSig, f_torch = lookup_torch_function(torch_fun_name)
    trace_x = generate_inputs_from_fun_sig(torchFunSig, dtype, device)[0] #assume arity one
    tvm_mod = torch_to_tvm_mod(torch_module_patch(torch_fun_name, f_torch), trace_x)
    tvm_mod = tvm_grad_gen(tvm_mod)

    f_tvm = lambda x: eval_tvm_mod_fun(tvm_mod, [x], fun='main', dtype=dtype, device=device)
    tvm_grad = lambda x: eval_tvm_mod_fun(tvm_mod, [x], fun='grad', dtype=dtype, device=device)
    tvm_grad2 = lambda x: eval_tvm_mod_fun(tvm_mod, [x], fun='grad2', dtype=dtype, device=device)
    return f_tvm, tvm_grad, tvm_grad2


def torch_to_tvm_mod(torch_fun, xs):
    # torch to typed relay
    torchscript = torch.jit.trace(torch_fun, xs)
    inputs = list(torchscript.graph.inputs())
    # shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in list(torchscript.graph.inputs())]
    shape_list = [("{}:{}".format(torch_fun.__name__, i), param.type().sizes()) for (i, param) in enumerate(inputs)]
    # print("shape list", shape_list)
    mod, _ = relay.frontend.from_pytorch(torchscript, shape_list)
    mod = transform.Sequential([relay.transform.InferType()])(mod) # infer types, needed for execution and gradient
    return mod


def tvm_grad_gen(mod):
    grad = tvm.relay.transform.gradient(mod['main'], mod=mod)
    mod['grad'] =   tvm.relay.Function(
                        grad.params,
                        tvm.relay.TupleGetItem(
                            tvm.relay.TupleGetItem(
                                grad.body,
                                1
                            ), 0)
                    )
    mod = transform.Sequential([relay.transform.InferType()])(mod)

    grad2 = tvm.relay.transform.gradient(mod['grad'], mod=mod)
    mod['grad2'] =  tvm.relay.Function(
                        grad2.params,
                        tvm.relay.TupleGetItem(
                            tvm.relay.TupleGetItem(
                                grad2.body,
                                1
                            ), 0)
                    )
    mod = transform.Sequential([relay.transform.InferType()])(mod)
    return mod


def eval_tvm_mod_fun(mod, xs, dtype='float32', device='cpu', fun='main'):
    # Note: not sure why, but if you don't typecheck here then running 'main' twice
    # results in 'function expected 1 argument, got 0' error
    mod = transform.Sequential([relay.transform.InferType()])(mod)
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
        ctx = tvm.cpu() if device == 'cpu' else tvm.gpu()
        target = tvm.target.arm_cpu() if device == 'cpu' else tvm.target.cuda()
        # Only CPU tensor can be converted to numpy
        # This is fine even for GPU, since the tvm evaluator will target the device parameter
        tvm_xs = [tvm.relay.const(tvm.nd.array(x.cpu().detach().numpy().astype(dtype), ctx)) for x in xs]

        #TODO: change execution model to Virtual Machine instead of interpreter by setting "kind='vm'"
        result = tvm.relay.create_executor(mod=mod, ctx=ctx, target=target).evaluate(
                        expr=tvm.relay.Call(mod[fun], tvm_xs))
        # result = tvm.relay.create_executor(mod=mod, ctx=ctx, target=target).evaluate(
        #                 mod.get_global_var(fun))(tvm_xs)
        return torch.from_numpy(result.asnumpy()).to(device)


def test_torch_fun_to_tvm(funName, dtype, device):
    print("Testing: '{}' (dtype={}, device={})".format(funName, dtype, device))

    print("looking up '{}' in PyTorch by name...".format(funName))
    torchFun = lookup_torch_func(funName)
    torchFunSig = lookup_torch_fun_sig(funName)
    if not torchFun:
        print("[[ ERROR ]]: failed to look up '{}' in PyTorch".format(funName))
        return (funName, 0, "could not look up in PyTorch", ""), False

    xs = None
    try:
        print("generating inputs...")
        xs = generate_inputs_from_fun_sig(torchFunSig, dtype, device)
        print("xs: ", xs)
    except Exception as e:
        print("[[ ERROR ]]: input generation failed for {}".format(funName))
        print(str(e))
        return (funName, 1, "input generation", str(e)), False


    mod = None
    try:
        print("converting to tvm module...")
        mod = torch_to_tvm_mod(torch_module_patch(funName, torchFun), xs)
    except Exception as e:
        print('[[ ERROR ]]: tvm function generation failed for {}'.format(funName))
        print(e)
        return (funName, 2, "missing op" if "operators are not implemented" in str(e) else "conversion failed", str(e)), False

    try:
        print("generating gradient...")
        mod = tvm_grad_gen(mod)
    except Exception as e:
        print('[[ ERROR ]]: grad generation failed for {}'.format(funName))
        print(e)
        return (funName, 3, "missing grad" if "MissingGrad" in str(e) else "grad gen failed", str(e)), False

    try:
        print("executing tvm-ified '{}'...".format(funName))
        eval_tvm_mod_fun(mod, xs, dtype, device, 'main')

    except Exception as e:
        print('[[ ERROR ]]: tvm-ified execution failed for "{}"'.format(funName))
        print(e)
        return (funName, 4, "tvm execution failed", str(e)), False


    try:
        print("executing 1st/2nd gradients...")
        eval_tvm_mod_fun(mod, xs, dtype, device, 'grad')
        eval_tvm_mod_fun(mod, xs, dtype, device, 'grad2')

    except Exception as e:
        print('[[ ERROR ]]: grad execution failed for {}'.format(funName))
        print(e)
        return (funName, 5, "missing op" if "The following operators are not implemented" in str(e) else
                             "missing grad" if "MissingGrad" in str(e) else
                             "grad eval failed", str(e)), False

    print("[[ SUCCESS ]]: '{}' (dtype={}, device={}) successfully passed the torch_to_tvm test".format(funName, dtype, device))
    return (funName, 6, "success"), True

def tvm_compatible_torch_funs():
    test = lambda fun_decl, dtype, device: test_torch_fun_to_tvm(fun_decl[0], dtype, device)
    blacklist = ['norm', 'pdist']
    filter_p = lambda fun_sig: fun_sig[0] not in blacklist and len(fun_sig[1].typs) == 2
    passes, fails = check_property_funs(test, filter_p) # Remember: .typs contains input type
    print("Testing finished.")
    return [p[1] for p in passes]


if __name__ == "__main__":
    import random
    enable_logger_debug()

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

    def generate_torch_derivatives(f):
        f_grad = lambda x: jacobian(f, x, create_graph=True)
        f_grad2 = lambda x: jacobian(f_grad, x)
        return f_grad, f_grad2

    # TODO: pass signature instead of function name
    def generate_tvm_derivatives(f_name, f, dtype, device):
        _, torchFunSig, _ = lookup_torch_function(f_name)
        x = generate_inputs_from_fun_sig(torchFunSig, dtype, device)[0] #assume arity one
        tvm_mod = torch_to_tvm_mod(torch_module_patch(f_name, f), x)
        tvm_mod = tvm_grad_gen(tvm_mod)

        tvm_f = lambda x: eval_tvm_mod_fun(tvm_mod, [x], fun='main',
                                           dtype=dtype, device=device)
        tvm_grad = lambda x: eval_tvm_mod_fun(tvm_mod, [x], fun='grad',
                                              dtype=dtype, device=device)
        tvm_grad2 = lambda x: eval_tvm_mod_fun(tvm_mod, [x], fun='grad2',
                                               dtype=dtype, device=device)

        return tvm_f, tvm_grad, tvm_grad2

    def torch_to_tvm_test(funName, dtype, device, compose=0):
        print("Testing equivalence of {} layers of 'sin' under compilation to tvm".format(compose+1))
        print("Using dtype:{} and device:{}".format(dtype, device))

        f = self_compose(compose, lookup_torch_func(funName))
        fun_sig = lookup_torch_fun_sig(funName)

        # generate inputs for testing
        xs = generate_inputs_from_fun_sig(fun_sig, dtype, device)

        # generate gradients for comparison test
        torch_grad, torch_grad2 = generate_torch_derivatives(f)
        tvm_f, tvm_grad, tvm_grad2 = generate_tvm_derivatives(funName, f,
                                                              dtype, device)

        # generate actual vs expected values
        test_pairs = [
                ("original function", f(*xs), tvm_f(*xs)),
                ("gradient", torch_grad(*xs), tvm_grad(*xs)),
                ("second derivative", torch_grad2(*xs), tvm_grad2(*xs))]

        # evaluate and compare actual (tvm) and expected (pytorch) values
        test_results = [(name,
                        actual, expected,
                        torch.isclose(actual, expected).all().item())
                        for (name, actual, expected) in test_pairs]

        test_result_output = "\n".join(
                ["[[ {} ]] Test '{}' :\ndiff: {}".format(
                    "SUCCESS" if result else "FAIL",
                 name,
                 expected-actual)
                 for name, actual, expected, result in test_results])

        assert all([result for _, _, _, result in test_results]), \
            "one or more tests failed. Results:\n{}".format(test_result_output)

        print(("[[ SUCCESS ]]: torch to tvm conversion test on {} '{}' layers"
              "passed for 0th, 1st, and 2nd derivatives")
              .format(compose, funName))


    def plot_function(ax, f, color, linestyle="solid", x_start=-100., x_end=100.):
        xs = [torch.tensor(float(x)) for x in range(int(x_start), int(x_end))]
        ys = [f(x) for x in xs]
        if ax is None:
            plt.plot(xs, ys, linestyle=linestyle, color=color)
        else:
            ax.plot(xs, ys, linestyle=linestyle, color=color)

    # Compares behavior of different layers of a torch function with its tvm-compiled equivalent
    def visualize_torch_vs_tvm(f, dtype, device, num_plots=4):
        def aggregate(f):
            return lambda x: f(x).sum()

        def plot_torch_vs_tvm(fun_name, f, dtype, device, compose=0):
            f = self_compose(compose, f)
            torch_grad, torch_grad2 = generate_torch_derivatives(f)
            tvm_f, tvm_grad, tvm_grad2 = generate_tvm_derivatives(fun_name, f, dtype, device)

            plot_function(None, aggregate(f), 'r')
            plot_function(None, aggregate(tvm_f), 'g', 'dashed')
            plot_function(None, lambda x: aggregate(f)(x) - aggregate(tvm_f)(x), 'b')

        fun_name = f.__name__
        fig, ax = plt.subplots()
        for i in range(num_plots):
            plot_torch_vs_tvm(fun_name, f, dtype, device, i)
        ax.set(title="1 to {} layers {} ({}, {})".format(num_plots, fun_name, dtype, device))
        plt.show()

    # Compares tvm-compiled versions of a layered torch function, at different layer depths
    def visualize_layers(f, dtype, device, num_plots=4):
        def aggregate(f):
            return lambda x: f(x).sum()

        def plot_composition(ax, fun_name, f, dtype, device, color, compose=0):
            f = self_compose(compose, f)
            tvm_f, _, _= generate_tvm_derivatives(fun_name, f, dtype, device)

            ax.set(title="{} layers {} ({}, {})".format(compose+1, fun_name, dtype, device))
            plot_function(ax, aggregate(tvm_f), color=color, linestyle='dotted')

        fun_name = f.__name__
        fig, ax = plt.subplots()
        for i in range(num_plots):
            color = (random.random(), random.random(), random.random())
            plot_composition(ax, fun_name, f, dtype, device, color, i)

        ax.grid(True)
        plt.show()

    def torch_to_tvm_fp32_representation_test(funName):
        # helpers
        import bitstring
        binary = lambda x: bitstring.BitArray(float=x, length=32).bin
        # tensor_to_npfloat32 = lambda x: np.float32(x.item())


        dtype = 'float32'
        torch_f = lookup_torch_func(funName)

        # generate inputs for testing
        x = torch.tensor(random.random()).to(torch.float32)

        # generate gradients for comparison test
        # torch_grad, torch_grad2 = generate_torch_derivatives(f)
        tvm_f_cpu, _, _= generate_tvm_derivatives(funName, torch_f, dtype, 'cpu')
        tvm_f_gpu, _, _= generate_tvm_derivatives(funName, torch_f, dtype, 'cuda:0')

        torch_cpu = torch_f(x.to('cpu')).to(torch.float32)
        torch_gpu = torch_f(x.to('cuda:0')).to(torch.float32)
        tvm_cpu = tvm_f_cpu(x).to(torch.float32)
        tvm_gpu = tvm_f_gpu(x).to(torch.float32)
        named_results = [
                ('torch_cpu', torch_cpu),
                ('torch_gpu', torch_gpu),
                ('tvm_cpu', tvm_cpu),
                ('tvm_gpu', tvm_gpu)]
        results = [res[1] for res in named_results]
        if not results.count(results[0]) == len(results):
            print("fp32 outputs differ at {}:".format(x.item()))
            print("\n".join(["{}\n{}".format(
                    res[0], binary(res[1].item())) for res in named_results]))
        else:
            print("fp32 outputs are identical.")
        print("normalized diffs:")
        named_results_normalized = [(res[0], torch.abs(torch_cpu - res[1]).item()) for res in named_results]
        print("\n".join(["{}: {}\t\t\t\t{}".format(res[0], res[1], binary(res[1])) for res in named_results_normalized]))


    def test_suite():
        torch_to_tvm_fp32_representation_test('linear')
        # torch_to_tvm_test('sin', dtype='float32', device='cpu')
        # torch_to_tvm_test('sin', dtype='float32', device='cuda:0')
        # torch_to_tvm_test('sin', dtype='float32', device='cpu', compose=1)
        # torch_to_tvm_test('sin', dtype='float32', device='cuda:0', compose=1)
        # torch_to_tvm_test('sin', dtype='float32', device='cpu', compose=5)
        # torch_to_tvm_test('sin', dtype='float32', device='cuda:0', compose=5)
        # test_torch_fun_to_tvm('sin', 'float32', 'cpu')
        # test_torch_fun_to_tvm('sin', 'float32', 'cuda:0')
        pass

    def view_torch_to_tvm_errors():
        test = lambda fun_decl, dtype, device: test_torch_fun_to_tvm(fun_decl[0], dtype, device)
        blacklist = ['norm', 'pdist']
        filter_p = lambda fun_sig: fun_sig[0] not in blacklist and len(fun_sig[1].typs) == 2
        passes, fails = check_property_funs(test, filter_p) # Remember: .typs contains input type
        print("Testing finished.")
        return passes, fails

    passes, fails = view_torch_to_tvm_errors()
    # test_suite()
    # funs = tvm_compatible_torch_funs()
    # visualize_torch_vs_tvm(torch.sin, 'float32', 'cpu')
    # visualize_layers(torch.sin, 'float32', 'cpu')
    # visualize_torch_vs_tvm(torch.sin, 'float32', 'cuda:0')
    # visualize_layers(torch.sin, 'float32', 'cuda:0')