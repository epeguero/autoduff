from autoduff_logging import log
import sys
import inspect
import torch
from torch.nn import *
from torch.nn.functional import *
from torch.nn.utils import *
import random
import copy
from itertools import chain
from functools import partial

import concurrent.futures

# constants
all_dtypes = {  'float16':torch.float16,
                'float32':torch.float32,
                'float64':torch.float64 }
all_devices = ['cpu', 'cuda:0']
all_sources = ['get_testing_overrides', 'scraper', 'manual']

# Base Types
class NumTyp():
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__

    def __str__(self):
        return "num"

class IntTyp():
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__

    def __str__(self):
        return "int"

# class FloatTyp():
#     def __eq__(self, other):
#         return self.__class__.__name__ == other.__class__.__name__
#
#     def __str__(self):
#         return "float"

class BoolTyp():
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__

    def __str__(self):
        return "bool"

class NoneTyp():
    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__

    def __str__(self):
        return "None"

class TensorTyp():
    def __init__(self, typ=NumTyp(), shape=(1,)): #shape=(2,2)):
        self.typ=typ
        self.shape = shape

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.typ == other.typ and self.shape == other.shape

    def __str__(self):
        return "Tensor({} {})".format(str(self.typ), "x".join(map(str, self.shape)))


# Compound Types
class FunTyp():
    def __init__(self, *typs):
        self.typs = typs

    def __str__(self):
        return " -> ".join(list(map(str, self.typs)))

    def __eq__(self, other):
        if len(self.typs) != len(other.typs):
            return False

        for t1, t2 in zip(self.typs, other.typs):
            if not t1 == t2:
                return False
        return True

class TupleTyp():
    def __init__(self, *typs):
        self.typs = typs

    def __str__(self):
        return "({})".format(", ".join(map(str, self.typs)))

class ListTyp():
    def __init__(self, typ):
        self.typ = typ

    def __str__(self):
        return "[{}]".format(str(self.typ))

class MaybeTyp():
    def __init__(self, typ):
        self.typ = typ

    def __str__(self):
        return str(self.typ) + "?"



# TODO: add functionality for uniformly sampling random numbers in the provided dtypes range
# currently, we sample from the interval [0, 1).
def gen_term_from_typ(typ, dtype, device):
    type_to_gen = type(typ)
    if type_to_gen is TensorTyp:
        found_nan = True
        generated_tensor = None
        while found_nan:
            if typ == TensorTyp(typ=IntTyp(), shape=typ.shape):
                generated_tensor = torch.rand(typ.shape, dtype=torch.int64).to(device)
                # generated_tensor = torch.zeros(typ.shape, dtype=torch.int64).to(device)
            elif typ == TensorTyp(typ=NumTyp(), shape=typ.shape):
                dtype_map = {'float16':torch.float16,
                          'float32':torch.float32,
                          'float64':torch.float64}
                generated_tensor = torch.rand(typ.shape, dtype=dtype_map[dtype]).to(device)
                # generated_tensor = torch.zeros(typ.shape, dtype=dtype_map[dtype]).to(device)
            elif typ == TensorTyp(typ=BoolTyp(), shape=typ.shape):
                generated_tensor = (torch.rand(typ.shape) > 0.5).to(device)
            else:
                raise Exception("unable to generate tensor of type {}".format(typ))
            found_nan = torch.isnan(generated_tensor).any()
            if found_nan:
                log.info("Generated tensor with nan... trying again")
        generated_tensor.detach()
        return generated_tensor.clone().squeeze()

    elif type_to_gen is NoneTyp:
        return None
    elif type_to_gen is NumTyp:
        return random.random()
    elif type_to_gen is IntTyp:
        return random.randint(0, 100)
    elif type_to_gen is BoolTyp:
        return random.random() > 0.5
    elif type_to_gen is MaybeTyp:
        return None
    elif type_to_gen is TupleTyp:
        return tuple(map(gen_term_from_typ, typ.typs))
    elif type_to_gen is ListTyp:
        return list(map(gen_term_from_typ, typ.typs))
    else:
        raise Exception("Failed to generate term for unrecognized type: ".format(typ_name))


def generate_inputs_from_fun_sig(fun_typ, dtype, device):
    input_typs = fun_typ.typs[0:-1]
    return list(map(lambda typ: gen_term_from_typ(typ, dtype, device), input_typs))


def typecheck_term(term, typ, dtype):
    actual_type = type(term)
    expected_type = type(typ)

    if actual_type is int:
        return expected_type is IntTyp
    elif actual_type is float:
        return expected_type is NumTyp
    elif actual_type is bool:
        return expected_type is BoolTyp
    elif actual_type is type(None):
        return expected_type is NoneTyp
    elif actual_type is tuple:
        if actual_type is TupleTyp:
            tuple_types = typ.typs
            if len(term) == len(tuple_types):
                tuple_typed_elems= zip(term, tuple_types)
                return all(map(lambda term, typ: typecheck_term(term, typ, dtype), *tuple_typed_elems))
            else:
                log.warning("typechecker expected {}-tuple, got {}-tuple instead".format(len(typ.typs), len(term)))
    elif actual_type is torch.Tensor:
        if expected_type is TensorTyp:
            expected_inner_type = type(typ.typ)
            # inner_typ_name = type(inner_typ).__name__.split('.')[-1]
            actual_shape = term.shape
            expected_shape = typ.shape
            dims = len(actual_shape)
            first_elem_ix = tuple([0 for _ in range(dims)])

            # torch flattens tuples into an additional dimension
            # thus, last dimension corresponds to tuple length
            if expected_inner_type is TupleTyp:
                actual_shape = term.shape[:-1]
                actual_tuple_len = term.shape[-1]
                expected_tuple_len = len(expected_inner_type.typs)
                # TODO: implement dtype check for tuple type
                if (actual_shape == expected_shape and
                    actual_tuple_len == expected_tuple_len):
                    first_elem_ix = first_elem_ix[:-1]
                    first_elem = term[first_elem_ix]
                    return typecheck_term(first_elem, typ.typ, dtype)
                else:
                    log.warning("Tuple Tensor shape mismatch: expected shape {} (got {}) and tuple length {} (got {}).".format(expected_shape, actual_shape, expected_tuple_len, actual_tuple_len))
            # assumes that if first element typechecks, the rest do as well
            elif actual_shape == expected_shape:
                def typecheck_dtype(typ, dtype):
                    expected_type = type(typ)
                    dtype_prefix = []
                    if expected_type is NumTyp:
                        dtype_prefix = ['float', 'int']
                    elif expected_type is IntTyp:
                        dtype_prefix = ['int']
                    elif expected_type is BoolTyp:
                        dtype_prefix = ['int', 'bool']
                    else:
                        log.warning("Unimplemented dtype for type: {}".format(typ))
                    return any(map(lambda pre: pre in str(dtype), dtype_prefix))

                if typecheck_dtype(typ.typ, term.dtype):
                    first_elem = term[first_elem_ix].item()
                    return typecheck_term(first_elem, typ.typ, dtype)
                else:
                    log.warning("Tensor dtype mismatch: actual dtype {} incompatible with expected autoduff type '{}'".format(term.dtype, typ.typ))
            else:
                log.warning("Tensor shape mismatch: expected shape {} (got {})".format(expected_shape, actual_shape))
        else:
            log.warning("Type mismatch: expected {} (got {})".format(typ, actual_type))
    else:
        log.warning("Unsupported term type {} passed to autoduff typechecker.".format(actual_type))

    return False


def typecheck_fun(fun_sig, fun, device, dtype):
    fun_name, fun_type_sig = fun_sig
    fun_sig_str = "{} :: {}".format(fun_name, fun_type_sig)
    log.info("Typechecking {} :: {}".format(fun_name, fun_type_sig))

    test_inputs = None
    try:
        test_inputs = generate_inputs_from_fun_sig(fun_type_sig, dtype, device)
    except Exception as e:
        log.error("[[FAIL]]: Error generating inputs for {}:\n\t{}".format(fun_sig_str, e))
        return False

    output = None
    try:
        output = fun(*test_inputs)
    except Exception as e:
        log.warning("[[FAIL]]: Error running test function {}({})".format(fun_name, test_inputs))
        log.warning("\t{}".format(e))
        return False

    try:
        output_typ = fun_type_sig.typs[-1]
        assert typecheck_term(output, output_typ, dtype), "[[ERROR]]: Type error in {} signature: expected output type {}, got {}".format(fun_name, output_typ, type(output))
    except AssertionError as e:
        log.error(str(e))
        if type(output_typ) == type(TensorTyp()) and type(output) == type(torch.Tensor()):
            log.error("\tExpected tensor shape {}, got {}".format(output_typ.shape, output.shape))

        log.error("\tFix the type signature! (Unless you merged many torch versions with different types)")
        return False

    else:
        log.info("[[SUCCESS]]: {} is well-typed".format(fun_name))
        return True


# helper functions for `arity_one_perms`
# needs to be defined in top-level for multiprocessing to work
def build_perm(fun, par_ix, args, param):
    perm_args = []
    perm_args.extend(args[0:par_ix])
    perm_args.append(param)
    perm_args.extend(args[par_ix+1:])
    return fun(*perm_args)

# output perms :: [(("name(par i)", type_sig), <python function>)]
def arity_one_perms(fun_sig, fun, dtype, device):
    fun_name, sig = fun_sig
    input_typs = sig.typs[:-1]
    output_typ = sig.typs[-1]

    arity = len(input_typs)
    if arity == 0:
        return []

    elif arity == 1:
        return [(fun_sig, fun)]

    perms = []
    args = list(map(lambda typ: gen_term_from_typ(typ, dtype, device), input_typs))
    for par_ix in range(1,arity):
        perm_typ = FunTyp(input_typs[par_ix], output_typ)
        perm = ("{}(par {})".format(fun_name, par_ix), perm_typ), partial(build_perm, fun, par_ix, args)
        perms.append(perm)
    return perms


def scalarize(a, b, f, *args):
    return torch.matmul(torch.matmul(a, f(*args)), b)

def scalarize_fun(fun_sig, fun, dtype, device):
    fun_name, sig = fun_sig
    input_typs = sig.typs[:-1]
    output_typ = sig.typs[-1]
    scalarized_fun_name = "{}_scalarized".format(fun_name)

    if type(output_typ) in [NumTyp, IntTyp]:
        return fun_name, fun_sig, None
    elif isinstance(output_typ, TensorTyp) and len(output_typ.shape) == 2:
        # args = list(map(lambda typ: gen_term_from_typ(typ, dtype, device), input_typs))
        # output = fun(*args)
        if not isinstance(output_typ, TensorTyp):
            raise Exception("[[ERROR]]: failed to scalarize {} due to bad output type{}".format(fun_name, sig))
        elif len(output_typ.shape) != 2:
            raise Exception("[[ERROR]]: failed to scalarize {} due to bad shape {} (only 2D output supported)".format(fun_name, output.shape))
        else:
            scalarized_typ_sig = FunTyp(*input_typs, TensorTyp(shape=tuple(), typ=output_typ.typ))
            v = torch.rand((1, output_typ.shape[0]), dtype=all_dtypes[dtype]).to(device).clone()
            w = torch.rand((output_typ.shape[-1], 1), dtype=all_dtypes[dtype]).to(device).clone()
            log.info("generated scalarized fun {} :: {}".format(scalarized_fun_name, scalarized_typ_sig))
            return ((scalarized_fun_name, scalarized_typ_sig), partial(scalarize, v, w, fun))#lambda *args: torch.matmul(torch.matmul(v, fun(*args)), w))
    return None



unary = FunTyp(TensorTyp(), TensorTyp())
binary = FunTyp(TensorTyp(), TensorTyp(), TensorTyp())
ternary = FunTyp(TensorTyp(), TensorTyp(), TensorTyp(), TensorTyp())
# in_place = FunTyp(Tensor(), NoneTyp())
scalar_tensor_shape = ()
scalar_tensor = TensorTyp(shape=scalar_tensor_shape)

pytorch_tensor_api = [
    ('abs', unary),
    # ('abs_', unary),
    ('absolute', unary),
    # ('absolute_', in_place),
    ('acos', unary),
    # ('acos_', unary),
    ('add', binary),
    # ('add_', in_place),
    ('asin', unary),
    ('atan', unary),
    ('atan2', FunTyp(TensorTyp(), TensorTyp(), TensorTyp())),
    ('addcdiv', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(), TensorTyp())),
    ('addcmul', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(), TensorTyp())),
    ('angle', FunTyp(TensorTyp(), TensorTyp())),
    ('bitwise_not', FunTyp(TensorTyp(BoolTyp()), TensorTyp(BoolTyp()))),
    ('bitwise_and', FunTyp(TensorTyp(BoolTyp()), TensorTyp(BoolTyp()), TensorTyp(BoolTyp()))),
    ('bitwise_or', FunTyp(TensorTyp(BoolTyp()), TensorTyp(BoolTyp()), TensorTyp(BoolTyp()))),
    ('bitwise_xor', FunTyp(TensorTyp(BoolTyp()), TensorTyp(BoolTyp()), TensorTyp(BoolTyp()))),
    ('ceil', unary),
    #TODO: requires positive semi-definite input matrix (ie, need to get lucky atm)
    ('clamp', FunTyp(TensorTyp(), NumTyp(), NumTyp(), TensorTyp())),
    ('conj', unary),
    ('cos', unary),
    ('cosh', unary),
    ('acosh', unary),
    # TODO: these require int arg to be dimension of input: consider adding default value to types
    # ('logcumsumexp', FunTyp(TensorTyp(), IntTyp(), TensorTyp())),
    # ('cumsum', FunTyp(TensorTyp(), IntTyp(), TensorTyp())),
    ('deg2rad', unary),
    #TODO: doesn't work on CPU backend
    # ('dequantize', unary),
    ('div', FunTyp(TensorTyp(), NumTyp(), TensorTyp())),
    ('digamma', unary),
    ('erf', unary),
    ('erfc', unary),
    ('erfinv', unary),
    ('exp', unary),
    ('expm', unary),
    ('floor', unary),
    ('floor_divide', binary),
    ('fmod', binary),
    ('fractional', unary),
    # TODO: only works over tensors with complex dtype, which we don't need atm
    # ('imag', unary),
    ('lerp', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(), TensorTyp())),
    ('lgamma', unary),
    ('log', unary),
    ('log10', unary),
    ('log1p', unary),
    ('log2', unary),
    ('logaddexp', binary),
    ('logaddexp2', binary),
    ('logical_and', FunTyp(TensorTyp(BoolTyp()), TensorTyp(BoolTyp()), TensorTyp(BoolTyp()))),
    ('logical_or', FunTyp(TensorTyp(BoolTyp()), TensorTyp(BoolTyp()), TensorTyp(BoolTyp()))),
    ('logical_xor', FunTyp(TensorTyp(BoolTyp()), TensorTyp(BoolTyp()), TensorTyp(BoolTyp()))),
    #TODO: expects values between 0 and 1 (probabilities)
    # ('bernoulli', FunTyp(TensorTyp(), TensorTyp()))
    ('mul', FunTyp(TensorTyp(), NumTyp(), TensorTyp())),
    ('mvlgamma', FunTyp(TensorTyp(), IntTyp(), TensorTyp())),
    ('neg', unary),
    ('polygamma', FunTyp(IntTyp(), TensorTyp(), TensorTyp())),
    ('pow', FunTyp(TensorTyp(), NumTyp(), TensorTyp())),
    ('rad2deg', unary),
    #TODO: only works on complex dtypes
    # ('real', unary),
    ('reciprocal', unary),
    ('remainder', FunTyp(TensorTyp(), NumTyp(), TensorTyp())),
    ('round', unary),
    ('rsqrt', unary),
    ('sigmoid', unary),
    ('sign', unary),
    ('sin', unary),
    ('sinh', unary),
    ('softsign', unary),
    ('sqrt', unary),
    ('square', unary),
    ('tan', unary),
    ('tanh', unary),
    ('tanhshrink', unary),
    ('normal', binary),
    ('true_divide', binary),
    ('trunc', unary),

    # Reduction Ops
    ('argmax', FunTyp(TensorTyp(), TensorTyp(shape=scalar_tensor_shape, typ=IntTyp()))),
    ('argmin', FunTyp(TensorTyp(), TensorTyp(shape=scalar_tensor_shape, typ=IntTyp()))),
    ('dist', FunTyp(TensorTyp(), TensorTyp(), scalar_tensor)),
    ('logsumexp', FunTyp(TensorTyp(), IntTyp(), NumTyp())),
    ('mean', FunTyp(TensorTyp(), scalar_tensor)),
    ('median', FunTyp(TensorTyp(), scalar_tensor)),
    ('mode', FunTyp(TensorTyp(), IntTyp(), BoolTyp(), TupleTyp(TensorTyp(), TensorTyp()))),
    ('norm', FunTyp(TensorTyp(), scalar_tensor)),
    ('prod', FunTyp(TensorTyp(), scalar_tensor)),
    ('std', FunTyp(TensorTyp(), scalar_tensor)),
    ('std_mean', FunTyp(TensorTyp(), TupleTyp(scalar_tensor, scalar_tensor))),
    ('sum', FunTyp(TensorTyp(), scalar_tensor)),
    # TODO: output type is the number of unique elements in input tensor
    # ('unique', FunTyp(TensorTyp(), TensorTyp()))
    # ('unique_consecutive', ___)
    ('var', FunTyp(TensorTyp(), scalar_tensor)),
    ('var_mean', FunTyp(TensorTyp(), TupleTyp(scalar_tensor, scalar_tensor))),

    # Comparison Ops
    ('allclose', FunTyp(TensorTyp(), TensorTyp(), NumTyp(), NumTyp(), BoolTyp(), BoolTyp())),
    ('argsort', FunTyp(TensorTyp(), TensorTyp(IntTyp()))),
    ('eq', FunTyp(TensorTyp(), NumTyp(), TensorTyp(BoolTyp()))),
    ('equal', FunTyp(TensorTyp(), TensorTyp(), BoolTyp())),
    ('ge', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(typ=BoolTyp()))),
    ('gt', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(typ=BoolTyp()))),
    ('isclose', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(typ=BoolTyp()))),
    ('isfinite', FunTyp(TensorTyp(), TensorTyp(typ=BoolTyp()))),
    ('isinf', FunTyp(TensorTyp(), TensorTyp(typ=BoolTyp()))),
    ('isnan', FunTyp(TensorTyp(), TensorTyp(typ=BoolTyp()))),
    # TODO: very weird dependent-type signature
    # ('kthvalue', )
    ('le', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(typ=BoolTyp()))),
    ('lt', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(typ=BoolTyp()))),
    # TODO: Note warning in docs about non-deterministic gradients for 'max/min'
    ('max', FunTyp(TensorTyp(), scalar_tensor)),
    ('min', FunTyp(TensorTyp(), scalar_tensor)),
    ('ne', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(typ=BoolTyp()))),
    ('sort', FunTyp(TensorTyp(), TupleTyp(TensorTyp(), TensorTyp(typ=IntTyp())))),
    # TODO: output shape depends on 'k' parameter
    # ('topk', __)

    # TODO: some of these functions need support for complex dtype
    # Spectral Ops
    # fft
    # ifft
    # rfft
    # irfft
    # stft
    # istft
    # bartlett_window
    # blackman_window
    # hamming_window
    # hann_window

    ('bincount', FunTyp(TensorTyp(), TensorTyp(typ=IntTyp()))),
    # TODO: block_diag takes an arbitrary number of input tensors, multiple type sigs can arise from here
    ('block_diag', FunTyp(TensorTyp(), TensorTyp())),
    ('broadcast_tensors', FunTyp(TensorTyp(), ListTyp(typ=TensorTyp()))),
    # TODO: bucketize requires that the second arg be monotonically increasing sequence
    ('bucketize', FunTyp(TensorTyp(), TensorTyp(shape=(1,)), TensorTyp(typ=IntTyp()))),
    #('cartesian_prod', FunTyp(TensorTyp(shape=(2,)), TensorTyp(shape=(2,)), TensorTyp(typ=TupleTyp((NumTyp(), NumTyp()))))),
    ('cdist', binary),
    # TODO: length is function of input size
    # ('combinations', FunTyp(TensorTyp(shape=(2,)), TensorTyp(shape=(2,), typ=TupleTyp((NumTyp(), NumTyp()))))),
    ('cross', FunTyp(TensorTyp(shape=(3,3,3)), TensorTyp(shape=(3,3,3)), TensorTyp(shape=(3,3,3)))),
    # TODO: these require int arg to be dimension of input: consider adding default value to types
    # ('cummax', FunTyp(TensorTyp(), IntTyp(), TupleTyp(TensorTyp(), TensorTyp()))),
    # ('cummin', FunTyp(TensorTyp(), IntTyp(), TupleTyp(TensorTyp(), TensorTyp()))),
    # ('cumprod', FunTyp(TensorTyp(), IntTyp(), TensorTyp())),
    ('diag', FunTyp(TensorTyp(shape=(2,)), TensorTyp(shape=(2,2)))),
    # TODO: diag_embed output shape depends on int dimension args
    # ('diag_embed', FunTyp(TensorTyp(), MaybeTyp(IntTyp()), MaybeTyp(IntTyp()), MaybeTyp(IntTyp()), TensorTyp())),
    # ('einsum', __) #requires expression string
    # ('flatten', 
    ('flip', unary),
    ('fliplr', unary),
    ('flipud', unary),
    ('rot90', unary),
    ('histc', unary),
    # ('meshgrid',  #shape dependent on num of params
    ('logcumsumexp', FunTyp(TensorTyp(), IntTyp(), TensorTyp())),
    ('renorm', FunTyp(TensorTyp(), NumTyp(), IntTyp(), NumTyp(), TensorTyp())),
    # ('repeat_interleave', #shape dependent on size of repeats arg
    ('roll', FunTyp(TensorTyp(), IntTyp(), TensorTyp())),
    # ('searchsorted', # non-trivial shapes
    # ('tensordot', #non-trivial shapes
    ('tril', FunTyp(TensorTyp(shape=(2,2)), TensorTyp(shape=(2,2)))),
    # ('tril_indices', FunTyp(IntTyp(), IntTyp(), Tensor(typ=IntTyp()))) # output shape depends on inputs
    ('triu', FunTyp(TensorTyp(shape=(2,2)), TensorTyp(shape=(2,2)))),
    # ('triu_indices', FunTyp(IntTyp(), IntTyp(), Tensor(typ=IntTyp()))) # output shape depends on inputs
    ('vander', FunTyp(TensorTyp(shape=(2,)), TensorTyp(shape=(2,2)))),
    ('view_as_real', FunTyp(TensorTyp(), TensorTyp())),
    # ('view_as_complex', FunTyp(TensorTyp(shape=(2,2)), TensorTyp(shape=(2,)))),

    # BLAS and LAPACK Operations
    ('addbmm', FunTyp(TensorTyp(shape=(3,3)), TensorTyp(shape=(3,3,3)), TensorTyp(shape=(3,3,3)), TensorTyp(shape=(3,3)))),
    ('addmm', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(), TensorTyp())),
    ('addmv', FunTyp(TensorTyp(shape=(2,)), TensorTyp(shape=(2,2)), TensorTyp(shape=(2,)), TensorTyp(shape=(2,)))),
    ('addr', FunTyp(TensorTyp(shape=(2,2)), TensorTyp(shape=(2,)), TensorTyp(shape=(2,)), TensorTyp(shape=(2,2)))),
    ('baddbmm', FunTyp(TensorTyp(shape=(2,2,2)), TensorTyp(shape=(2,2,2)), TensorTyp(shape=(2,2,2)), TensorTyp(shape=(2,2,2)))),
    ('bmm', FunTyp(TensorTyp(shape=(2,2,2)), TensorTyp(shape=(2,2,2)), TensorTyp(shape=(2,2,2)))),
    ('chain_matmul', binary),
    ('cholesky', FunTyp(TensorTyp(), BoolTyp(), TensorTyp())),
    ('cholesky_inverse', FunTyp(TensorTyp(), BoolTyp(), TensorTyp())),
    ('cholesky_solve', binary),
    ('dot', FunTyp(TensorTyp(shape=(2,)), TensorTyp(shape=(2,)), scalar_tensor)),

    ('eig', FunTyp(TensorTyp(), MaybeTyp(BoolTyp()), TupleTyp(TensorTyp(), TensorTyp()))),
    # ('geqrf', FunTyp  # from docs: "You'll generally want to use torch.qr() instead"
    ('inverse', unary),
    ('det', FunTyp(TensorTyp(), TensorTyp(shape=torch.Size([])))),
    ('logdet', FunTyp(TensorTyp(), TensorTyp(shape=torch.Size([])))),
    # ('slogdet', FunTyp(TensorTyp(), TensorTyp(shape=torch.Size([]), typ=TupleTyp(IntTyp()))),  # weird return type object
    ('lstsq', binary),
    # ('lu',  # returns tuple of decomposition
    # ('lu_solve',  # returns tuple of decomposition
    # ('lu_unpack',  # returns tuple of decomposition
    ('matmul', binary),
    ('matrix_power', FunTyp(TensorTyp(), IntTyp(), TensorTyp())),
    ('matrix_rank', FunTyp(TensorTyp(), IntTyp(), TensorTyp(shape=torch.Size([]), typ=IntTyp()))),
    ('mm', binary),
    ('mv', FunTyp(TensorTyp(shape=(2,2)), TensorTyp(shape=(2,)), TensorTyp(shape=(2,)))),
    ('orgqr', binary),
    ('ormqr', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(), TensorTyp())),
    ('pinverse', unary),
    # ('qr'  #returns tuple
    # ('solve', #returns tuple
    # 'svd'
    # 'svd_lowrank'
    # 'pca_lowrank'
    # 'symeig'
    # 'lobpcg'
    # 'trapz'
    # 'triangular_solve'

    # From nn.functional:
    # Convolution Functions
    ('conv1d', FunTyp(TensorTyp(shape=(2,2,2)), TensorTyp(shape=(2,2,2)), TensorTyp(shape=(2,2,1)))),
    ('conv2d', FunTyp(TensorTyp(shape=(2,2,2,2)), TensorTyp(shape=(2,2,2,2)), TensorTyp(shape=(2,2,1,1)))),
    ('conv3d', FunTyp(TensorTyp(shape=(2,2,2,2,2)), TensorTyp(shape=(2,2,2,2,2)), TensorTyp(shape=(2,2,1,1,1)))),

    ('conv_transpose1d', FunTyp(TensorTyp(shape=(2,2,2)), TensorTyp(shape=(2,2,2)), TensorTyp(shape=(2,2,3)))),
    ('conv_transpose2d', FunTyp(TensorTyp(shape=(2,2,2,2)), TensorTyp(shape=(2,2,2,2)), TensorTyp(shape=(2,2,3,3)))),
    ('conv_transpose3d', FunTyp(TensorTyp(shape=(2,2,2,2,2)), TensorTyp(shape=(2,2,2,2,2)), TensorTyp(shape=(2,2,3,3,3)))),

    # Pooling Functions
    ('avg_pool1d', FunTyp(TensorTyp(shape=(2,2,2)), IntTyp(), TensorTyp(shape=(2,1,0)))),
    ('avg_pool2d', FunTyp(TensorTyp(shape=(2,2)), TensorTyp())),
    ('avg_pool3d', FunTyp(TensorTyp(shape=(2,2,2)), TensorTyp())),
    ('max_pool1d', FunTyp(IntTyp(), IntTyp(), TensorTyp(shape=(10,)))),
    ('max_pool2d', FunTyp(IntTyp(), IntTyp(), TensorTyp(shape=(10,10)))),
    ('max_pool3d', FunTyp(IntTyp(), IntTyp(), TensorTyp(shape=(10,10,10)))),
    # 'max_unpool1d / 2d / 3d
        
    ('lp_pool1d', FunTyp(IntTyp(), IntTyp(), TensorTyp(shape=(10,)))),
    ('lp_pool2d', FunTyp(IntTyp(), IntTyp(), TensorTyp(shape=(10,10)))),

    # adaptive_max_pool1d / 2d / 3d
    # adaptive_avg_pool1d / 2d / 3d

    ('threshold', FunTyp(TensorTyp(), NumTyp(), NumTyp(), TensorTyp())),
    ('relu', unary),
    ('hardtanh', unary),
    ('hardswish', unary),
    ('relu6', unary),
    ('elu', unary),
    ('selu', unary),
    ('celu', unary),
    ('leaky_relu', unary),
    ('prelu', binary),
    ('rrelu', unary),
    ('glu', unary),
    ('gelu', unary),
    #('logsigmoid', unary),
    ('hardshrink', unary),
    ('softmin', unary),
    ('softmax', unary),
    #('softshrink', unary),
    #('gumbel_softmax', unary),
    ('log_softmax', unary),
    ('sigmoid', unary),
    ('hardsigmoid', unary),


    # Normalization Functions
    ('batch_norm', FunTyp(TensorTyp(), NumTyp(), NumTyp())),
    # ('instance_norm', FunTyp(TensorTyp())),
    ('layer_norm', binary),
    ('local_response_norm', FunTyp(TensorTyp(), IntTyp())),
    ('normalize', unary),

    # Linear Functions
    ('linear', binary),
    ('bilinear', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(), TensorTyp(), TensorTyp())),

    # Dropout Functions
  #  ('dropout', unary),
  #  ('alpha_dropout', unary),
  #  ('feature_alpha_dropout', unary),
  #  ('drouput2d', unary),
  #  ('drouput3d', unary),
  #
  #  # Sparse Functions
  #  ('embedding', binary),
  #  ('embedding_bag', binary),
  #  ('one_hot', unary),

    # Distance Functions
    ('pairwise_distance', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(shape=(2,)))),
    ('cosine_similarity', FunTyp(TensorTyp(), TensorTyp(), TensorTyp(shape=(2,)))),
    ('pdist', FunTyp(TensorTyp(), TensorTyp(shape=(2,)))),

    # Loss Functions
   # ('binary_cross_entropy', binary),
   # ('binary_cross_entropy_with_logits', binary),
   # ('poisson_nll_loss', binary),
   # ('cosine_embedding_loss', ternary),
   # ('cross_entropy', binary),
   # #('ctc_loss', 
   # ('hinge_embedding_loss', binary),
   # ('kl_div', binary),
   # ('l1_loss', binary),
   # ('mse_loss', binary),
   # #('margin_ranking_loss', FunTyp
   # ('multilabel_margin_loss', binary),
   # ('multilabel_soft_margin_loss', binary),
   # ('multi_margin_loss', binary),
   # ('nll_loss', binary), 
   # ('smooth_l1_loss', binary),
   # ('soft_margin_loss', binary),
   # ('triplet_margin_loss', ternary),
   # 
   # # Vision Functions
   # ('pixel_shuffle', unary),
   # #('pad', 
   # #('interpolate'
   # #'upsample'
   # ('upsample_nearest', unary),
   # ('upsample_bilinear', unary),
   # ('grid_sample', binary),
   # #'affine_grid'
   #
   # #'data_parallel'
]

ignored = [
    'cpu',
    'cuda',
    'data_ptr',
    'dense_dim',
    'detach',
    'detach_',
    'fill_diagonal_',
    'bool',
    'byte',
    'double',
    'element_size'
]

def lookup_torch_fun_sig(funName):
    return next(funSig for (funNameInList, funSig) in pytorch_tensor_api if funName == funNameInList)

def lookup_torch_func(fun_name):
    if fun_name == 'sigmoid':
        return torch.sigmoid
    if hasattr(torch.nn.functional, fun_name):
        return getattr(torch.nn.functional, fun_name)
    elif hasattr(torch, fun_name):
        return getattr(torch, fun_name)
    else:
        log.warning("[[ERROR]]: function {} not found in torch version {}".format(fun_name, torch.__version__))
        return None

def lookup_torch_function(funName):
    # (name, sig, fun)
    return next((apiFun[0], apiFun[1], lookup_torch_func(apiFun[0])) for apiFun in pytorch_tensor_api if apiFun[0] == funName)

# fun signature :: [(("name", type_sig), <python function>)]
def convert_all_to_single_arity(dtype='float32', device='cuda:0', compute_perms=True):
    flatten = lambda list_of_lists: [elem for l in list_of_lists for elem in l]

    version_compatible_funs = filter(lambda x: x is not None, [((fun_name, fun_typ_sig), lookup_torch_func(fun_name)) for fun_name, fun_typ_sig in pytorch_tensor_api])
    well_typed_funs = [fun_sig for fun_sig in version_compatible_funs if typecheck_fun(*fun_sig, device, dtype)]
    #scalarized_funs = filter(lambda x: x is not None, [scalarize_fun(*fun_sig, dtype, device) for fun_sig in well_typed_funs])
    #tensor_perms = flatten(map(lambda fun_sig: arity_one_perms(*fun_sig, dtype, device), chain(well_typed_funs,scalarized_funs)))
    if compute_perms:
        tensor_perms = flatten(map(lambda fun_sig: arity_one_perms(*fun_sig, dtype, device), well_typed_funs))
        return well_typed_funs + tensor_perms
    else:
        return well_typed_funs


def filter_torch_by(p):
    return [fun_sig for fun_sig in pytorch_tensor_api if p(fun_sig)]


def single_arity_funs():
    # Remember: input type is included in `typs`
    return filter_torch_by(lambda fun_sig: len(fun_sig[1].typs) == 2)


# property p:
#   function of type funDecl-> dtype -> device -> (arbitrary object, bool),
#   where bool denotes pass/fail, and arbitrary object collects results
# function filter_p: filter property used to specify set of functions to test (defaults to all)
def check_property_funs(p, filter_p=None):
    passes = []
    fails = []
    all_funs = pytorch_tensor_api if filter_p is None else filter_torch_by(filter_p)
    for dtype in all_dtypes:
        for device in all_devices:
            for fun_decl in all_funs:
                out, passed = p(fun_decl, dtype, device)
                dest = passes if passed else fails
                dest.append((out, (fun_decl[0], dtype, device)))

    return passes, fails

def check_property_funs_par(p, filter_p=None):
    passes = []
    fails = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        all_funs = pytorch_tensor_api if filter_p is None else filter_torch_by(filter_p)
        jobs = []
        for dtype in all_dtypes:
            for device in all_devices:
                for fun_decl in all_funs:
                    print('starting future to check property for ({}, {}, {})...'.format(fun_decl[0], dtype, device))
                    futures_dict = { executor.submit(p, fun_decl, dtype, device) : (fun_decl[0], dtype, device) }
                    for completed_future in concurrent.futures.as_completed(futures_dict):
                        fun_name, dtype, device = futures_dict[completed_future]
                        out, passed = completed_future.result()
                        print('future for ({}, {}, {}) completed.'.format(fun_name, dtype, device))
                        dest = passes if passed else fails
                        dest.append((out, (fun_decl[0], dtype, device)))

    return passes, fails

if __name__ == "__main__":
    def test_template(test_name, test):
        passes, fails = check_property_funs(test)
        print(passes)
        print("'{}' passes for {} functions.".format(test_name, len(passes)))
        print(fails)
        print("'{}' fails for {} functions.".format(test_name, len(fails)))

    def lookup_test():
        def test(fun_decl, _, __):
            fun_name = fun_decl[0]
            f = lookup_torch_func(fun_name)
            return f, (f is not None)
        test_template('lookup by name', test)

    def input_generation_test():
        def test(fun_decl, dtype, device):
            try:
                xs = generate_inputs_from_fun_sig(fun_decl[1], dtype, device)
                return xs, bool(xs)
            except Exception as e:
                return e, False
        test_template('input generation', test)

    # def wholistic_test():
    #     def test(fun_decl, dtype, device):
    #         funName = fun_decl[0]
    #         print("Testing: '{}' (dtype={}, device={})".format(funName, dtype, device))
    #
    #         print("looking up '{}' in PyTorch by name...".format(funName))
    #         torchFun = lookup_torch_func(funName)
    #         torchFunSig = lookup_torch_fun_sig(funName)
    #         if not torchFun:
    #             print("[[ ERROR ]]: failed to look up '{}' in PyTorch".format(funName))
    #             return (funName, 0, "could not look up in PyTorch", ""), False
    #
    #         xs = None
    #         try:
    #             print("generating inputs...")
    #             xs = generate_inputs_from_fun_sig(torchFunSig, dtype, device)
    #             # print("xs: ", xs)
    #         except Exception as e:
    #             print("[[ ERROR ]]: input generation failed for {}".format(funName))
    #             traceback.print_exc(file=sys.stdout)
    #             # if dtype != 'float16':
    #             #     raise
    #             return (funName, 1, "input generation", str(e)), False
    #
    #
            # mod = None
            # try:
            #     print("converting to tvm module...")
            #     mod = torch_to_tvm_mod(torch_module_patch(funName, torchFun), xs)
            # except Exception as e:
            #     print('[[ ERROR ]]: tvm function generation failed for {}'.format(funName))
            #     print(e)
            #     return (funName, 2, "missing op" if "operators are not implemented" in str(e) else "conversion failed", str(e)), False
            #
            # try:
            #     print("generating gradient...")
            #     mod = tvm_grad_gen(mod)
            # except Exception as e:
            #     print('[[ ERROR ]]: grad generation failed for {}'.format(funName))
            #     print(e)
            #     return (funName, 3, "missing grad" if "MissingGrad" in str(e) else "grad gen failed", str(e)), False
            #
            # try:
            #     print("executing tvm-ified '{}'...".format(funName))
            #     eval_tvm_mod_fun(mod, xs, dtype, device, 'main')
            #
            # except Exception as e:
            #     print('[[ ERROR ]]: tvm-ified execution failed for "{}"'.format(funName))
            #     print(e)
            #     return (funName, 4, "tvm execution failed", str(e)), False
            #
            #
            # try:
            #     print("executing 1st/2nd gradients...")
            #     eval_tvm_mod_fun(mod, xs, dtype, device, 'grad')
            #     eval_tvm_mod_fun(mod, xs, dtype, device, 'grad2')
            #
            # except Exception as e:
            #     print('[[ ERROR ]]: grad execution failed for {}'.format(funName))
            #     print(e)
            #     return (funName, 5, "missing op" if "The following operators are not implemented" in str(e) else
            #                          "missing grad" if "MissingGrad" in str(e) else
            #                          "grad eval failed", str(e)), False
            #
            # print("[[ SUCCESS ]]: '{}' (dtype={}, device={}) successfully passed the torch_to_tvm test".format(funName, dtype, device))
            # return (funName, 6, "success"), True
        # test_template('wholistic test', test)


    def test_suite():
        lookup_test()
        input_generation_test()

    test_suite()

    # torch_test_funs = { dtype:{} for dtype in all_dtypes }
    # for dtype in all_dtypes:
    #     for device in all_devices:
    #         torch_test_funs[dtype][device] = arity_one_funs(dtype, device)
