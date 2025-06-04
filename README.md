# GeMMMapReduce

This repo is a Proof of Concept of a general framework to construct and generalize 
memory efficient layers in neural networks. The idea is the same as Map Reduce, 
or monoidal folds, namely that we identify a mapping from inputs to monoidal 
values that when folded results in an aggregate from which the output can be
produced. 

A notable example is attention, where the monoid corresponds to logarithmic
weights and vectors representing a weighted average.

The repo contains a factory function `mk_GeMMMapReduce` which takes functions `init`, `chunker`, `proj_fold`, 
`proj_fold_bwd`, and `binary_reduce`, which performs the monoidal aggregation of results without realizing the full intermediate matrix needed
to compute the aggregate, and a similar backwards pass. As well as implementations for two-layer MLPs, Attention, XEntropy, Entropy, and Sampling.

```
def binary_reduce(a, b):
    # a, b : Attention Monoids represented as tuples of tensors
    # returns: "product" of a and b.
    # z: logarithmic weight (M-tensor)
    # v: weighted average (M x D-tensor)
    a_z, a_v = a
    b_z, b_v = b
    z = torch.logaddexp(a_z, b_z)
    v = a_v * torch.exp(a_z - z)[:, None] + b_v * torch.exp(b_z - z)[:, None]
    return z, v

def proj_fold(query, key, value):
    # Fused projection and fold.
    # Applied to slices over inputs.

    # query: M x F
    # key  : N x F
    # value: N x D
    # returns (M, M x D)

    zs = query @ key.T
    z = zs.logsumexp(dim=1)
    v = (zs - z[:, None]).exp() @ value
    return z, v

def proj_fold_bwd(query, key, value, a, ga):
    # (local) backwards pass for proj_fold.
    # using the property that d fold(X) / d X[i]
    # can be expressed as a function of fold(X) and X[i]. 

    # query, key, and value are slices of original inputs.
    # a : an output slice (fold(X)).
    # ga: an output gradient slice. 

    h_z = query @ key.T
    h_v = value

    a_z, a_v = a
    g_z, g_v = g

    w = (h_z - a_z[:, None]).exp()

    z = (g_z[:, None] + g_v @ h_v.T - (g_v * a_v).sum(1, keepdims=True)) * w
    v = w.T @ g_v
    return z @ key, z.T @ query, v

def init(query, key, value):
    # Constructs initial values for the accumulator.

    assert (query.shape[1] == key.shape[1])
    assert (key.shape[0] == value.shape[0])
    M, F = query.shape
    N, D = value.shape
    identity = query.new_full((M,), float('-inf')), query.new_zeros((M,D))
    return identity

def chunker(query, key, value):
    # Constructs iterator of output and input chunking functions.

    assert (query.shape[1] == key.shape[1])
    assert (key.shape[0] == value.shape[0])
    M, F = query.shape
    N, D = value.shape
    mslices = list(slicer(M, 256))
    nslices = list(slicer(N, 256))
    for mslice, nslice in product(mslices, nslices):
        yield (
            lambda A: (A[0][mslice], A[1][mslice]),
            lambda X: (X[0][mslice], X[1][nslice], X[2][nslice])
        )

Attention = mk_GeMMMapReduce(
        'Attention',
        init=init,
        chunker=chunker,
        proj_fold=proj_fold,
        proj_fold_bwd=proj_fold_bwd,
        binary_reduce=binary_reduce,
        )

def gemmmr_attention(q, k, v):
    z, v = Attention.apply(q, k, v)
    return v

def regular_attention(q, k, v):
    return (q @ k.T).softmax(1) @ v
```

To compute the backwards pass, we depend on the property that for some commutative monoids, `d fold(X) / d X[i]` can be expressed
as a function of `fold(X) and X[i]`. See the [paper](paper.pdf) for details.

The functions produced by `mk_GeMMMapReduce`, while sometimes proving more efficient than their regular torch counterparts on CPUs, are not
particularly competetive when running on GPUs. This repo serversas a proof-of-concept of the underlying idea. A possible path forward
is adapting the framework to work with user-provided triton kernels, rather than pytorch functions.
