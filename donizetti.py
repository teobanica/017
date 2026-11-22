# Weingarten calculus for the free orthogonal quantum group O_N^+:
# noncrossing pairings NC_2(k), joins, Gram and Weingarten matrices.
from fractions import Fraction
from functools import lru_cache
import numpy as np

def nc2(k):
    """Noncrossing pairings of {0,...,k-1}, as tuples of block labels."""
    if k == 0:
        return [()]
    res = []
    for j in range(1, k, 2):
        for a in nc2(j - 1):
            for b in nc2(k - j - 1):
                lab = [0] * k
                lab[j] = 0
                for t, x in enumerate(a):
                    lab[1 + t] = 1 + x
                off = 1 + (max(a) + 1 if a else 0)
                for t, x in enumerate(b):
                    lab[j + 1 + t] = off + x
                res.append(tuple(lab))
    return res

def join_blocks(p, q):
    """Number of blocks of the join of two partitions given by labels."""
    k = len(p)
    par = list(range(k))
    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for lab in (p, q):
        first = {}
        for i, b in enumerate(lab):
            if b in first:
                par[f(i)] = f(first[b])
            else:
                first[b] = i
    return len({f(i) for i in range(k)})

def interval(k):
    """The pairing {1,2}{3,4}..., used for the products of the p_ij."""
    return tuple(i // 2 for i in range(k))

def nested4():
    """The nested pairing {1,4}{2,3}."""
    return (0, 1, 1, 0)

@lru_cache(None)
def weingarten_float(k, N):
    P = nc2(k)
    G = np.array([[float(N) ** join_blocks(a, b) for b in P] for a in P])
    return P, np.linalg.inv(G)

def solve_exact(G, b):
    n = len(G)
    A = [row[:] + [b[i]] for i, row in enumerate(G)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c] / A[c][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return [A[i][n] / A[i][i] for i in range(n)]
