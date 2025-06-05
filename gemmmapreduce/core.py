import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import itertools
import time
from contextlib import contextmanager

def slicer(tot, chunk=1):
    start = 0
    while (start < tot):
        end = min(tot, start + chunk)
        yield slice(start, end)
        start = end

def mk_GeMMMapReduce(
    class_name,
    *,
    init,
    chunker,
    proj_fold,
    proj_fold_bwd,
    binary_reduce,
    ):
    class DynamicFunction(torch.autograd.Function):
        @staticmethod
        def forward(*X):
            A = init(*X)
            for a_views, x_views in chunker(*X):
                a = [Ai[*slz] for (Ai, slz) in zip(A, a_views)]
                x = [Xi[*slz] for (Xi, slz) in zip(X, x_views)]
                local_a = proj_fold(*x)
                new_a = binary_reduce(a, local_a)
                for view, new_val in zip(a, new_a):
                    view.copy_(new_val)
            return A
    
        @staticmethod
        def setup_context(ctx, inputs, outputs):
            ctx.num_inputs = len(inputs)
            ctx.save_for_backward(*inputs, *outputs)
    
        @staticmethod
        @once_differentiable
        def backward(ctx, *gA):
            X = ctx.saved_tensors[:ctx.num_inputs]
            A = ctx.saved_tensors[ctx.num_inputs:]
            gX = [p.new_zeros(p.shape) for p in X]
            for a_views, x_views in chunker(*X):
                # Extract chunks
                a = [Ai[*slz] for (Ai, slz) in zip(A, a_views)]
                ga = [gAi[*slz] for (gAi, slz) in zip(gA, a_views)]
                x = [Xi[*slz] for (Xi, slz) in zip(X, x_views)]
                gx_acc = [gXi[*slz] for (gXi, slz) in zip(gX, x_views)]
                # recompute and calculate local gradients.
                gx = proj_fold_bwd(*x, a, ga)
                # Add local gradients to global.
                for g, d in zip(gx_acc, gx):
                    g.add_(d)
            return tuple(gX)
    
    DynamicFunction.__name__ = class_name
    DynamicFunction.__qualname__ = class_name
    DynamicFunction.__module__ = getattr(init, '__module__', __name__)
    
    return DynamicFunction


def check_equality(f1, f2, inputs, mock):
    for p in inputs:
        if p.grad is not None:
            p.grad.zero_()
    y1 = f1(*inputs)
    (y1 * mock).sum().backward()
    y1 = y1.detach()
    g1 = []
    for p in inputs:
        if p.grad is not None:
            g1.append(p.grad.detach().clone())
    for p in inputs:
        if p.grad is not None:
            p.grad.zero_()
    y2 = f2(*inputs)
    (y2 * mock).sum().backward()
    y2 = y2.detach()
    g2 = []
    for p in inputs:
        if p.grad is not None:
            g2.append(p.grad.detach().clone())
    
    def check_pair(a, b):
        delta = a - b
        shapes_match = a.shape == b.shape
        all_close = torch.allclose(a, b)
        l2_diff = delta.pow(2).sum().sqrt()
        max_diff = delta.abs().max()
        print(f'{" shapes match": <20}: {shapes_match}')
        print(f'{" all close": <20}: {all_close}')
        print(f'{" l2 diff": <20}: {l2_diff}')
        print(f'{" max_diff": <20}: {max_diff}')
        print()
        if all_close and shapes_match:
            print('   All good! :)')
        else:
            print('   Something is wrong. :(')
        print()


    print(f'{" output ":=^30}')
    check_pair(y1, y2)


    print(f'{" grad ":=^30}')
    for i, (a, b) in enumerate(zip(g1, g2)):
        name = f' grad_{i} '
        print(f'  {name:-^26}  ')
        check_pair(a, b)


def check_speed(f1, inputs, mock, runs=10, warmup=3):
    for i in range(warmup):
        for p in inputs:
            if p.grad is not None:
                p.grad.zero_()
        y1 = f1(*inputs)
        (y1 * mock).sum().backward()

    start = time.perf_counter()
    for i in range(runs):
        acc = []
        y1 = f1(*inputs)
        (y1 * mock).sum().backward()
        with torch.no_grad():
            for p in inputs:
                if p.grad is not None:
                    acc.append(p.grad.sum())
        acc = sum(acc).item()

    return (time.perf_counter() - start) / runs
    
def check(f1, f2, inputs, mock, runs=10, warmup=3):
    check_equality(f1, f2, inputs, mock)
    print(f'{" speed ":=^30}')
    print(f' {" f1 ":-^26}  ')
    s1 = check_speed(f1, inputs, mock, runs, warmup)
    print(f' {s1:.2f}')
    print(f' {" f2 ":-^26}  ')
    s2 = check_speed(f2, inputs, mock, runs, warmup)
    print(f' {s2:.2f}')
    ratio = (s1 / s2)
    print(f' relative time: {ratio:2f}')
    if ratio > 1:
        print(f'   f1 is slower :(')
    else:
        print(f'   f1 is faster :)')
