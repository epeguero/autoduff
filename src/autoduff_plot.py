import matplotlib
import matplotlib.pyplot as plt

def compute_points(f, x_center, n):
    xs = [float(x) for x in range(int(x_center - n/2), int(x_center + n/2))]
    ys = [f(torch.tensor(x)).item() for x in xs]
    assert len(xs) >= n-2
    return xs, ys

def plot_function(f, color='b', linestyle="solid", x_center=0., n=20, ax=None):
    xs, ys = compute_points(f, x_center, n)
    min_x, max_x = min(xs), max(xs)
    graph = plt if ax is None else ax
    graph.plot(xs, ys, linestyle=linestyle, color=color)
    graph.xlim([min_x, max_x])

def scatter_function(f, color='b', x_center=0., n=20, ax=None):
    xs, ys = compute_points(f, x_center, n)
    min_x, max_x = min(xs), max(xs)
    graph = plt if ax is None else ax
    graph.scatter(xs, ys, color=color)
    graph.xlim([min_x, max_x])

def plot_torch_vs_tvm(fun_name, dtype, device, x_center=0.):
    f = lambda x: lookup_torch_func(fun_name)(x.to(all_dtypes[dtype]).to(device)).to('cpu')
    torch_grad, torch_grad2 = generate_torch_derivatives(f)
    tvm_f, tvm_grad, tvm_grad2 = generate_tvm_derivatives(fun_name, f, dtype, device)

    plt = scatter_function
    plt(f, 'r', x_center=x_center)
    plt(tvm_f, 'g', x_center=x_center)
    plt(lambda x: f(x) - tvm_f(x), 'b', x_center=x_center)

def plot_points(xs, ys, color='b', ax=None):
    graph = plt if ax is None else ax
    graph.scatter(xs, ys, color=color)

def plot_finish(name=None, save=False, show=True):
    if save:
        plt.save_fig(name)
    if show:
        plt.show()

