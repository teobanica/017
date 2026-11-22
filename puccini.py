# Numerical check of Theorem 7.4 at N=3, L=2: the nested invariants
#   X_abcd = sum_{r,s<=L} u_ra u_sb u_sc u_rd
# are not in the span of the products of at most 3 entries of p = u^t u,
# inside C(O_N^+). We compute Gram matrices with respect to the Haar state,
# via the Weingarten formula. Results go to puccini.txt.
import itertools
import numpy as np
from donizetti import interval, nested4, join_blocks, weingarten_float

N, L, M = 3, 2, 3

def concat_rev(t1, t2):
    rev = tuple(t1[::-1])
    m = max(rev) + 1 if rev else 0
    return rev + tuple(m + x for x in t2)

def gram(elems):
    # element (tau, i) = sum over r constant on blocks of tau of u_{r1 i1}...u_{rk ik}
    n = len(elems)
    G = np.zeros((n, n))
    groups = {}
    for e, (tau, idx) in enumerate(elems):
        groups.setdefault(tau, []).append(e)
    for t1, E1 in groups.items():
        for t2, E2 in groups.items():
            k = len(t1) + len(t2)
            P, W = weingarten_float(k, N)
            T = concat_rev(t1, t2)
            c = np.array([float(L) ** join_blocks(pi, T) for pi in P]) @ W
            I1 = np.array([elems[e][1][::-1] for e in E1])
            I2 = np.array([elems[e][1] for e in E2])
            J = np.concatenate([np.repeat(I1[:, None, :], len(E2), 1),
                                np.repeat(I2[None, :, :], len(E1), 0)], axis=2)
            B = np.zeros((len(E1), len(E2)))
            for s, sig in enumerate(P):
                ok = np.ones((len(E1), len(E2)), bool)
                blocks = {}
                for pos, b in enumerate(sig):
                    blocks.setdefault(b, []).append(pos)
                for x, y in blocks.values():
                    ok &= (J[:, :, x] == J[:, :, y])
                B += c[s] * ok
            G[np.ix_(E1, E2)] = B
    return G

def rank(G):
    s = np.linalg.svd(G, compute_uv=False)
    return int((s > 1e-8 * s[0]).sum())

out = open("puccini.txt", "w")
def say(s):
    print(s); out.write(s + "\n"); out.flush()

idx = lambda k: list(itertools.product(range(N), repeat=k))
X = [(nested4(), i) for i in idx(4)]
for m in range(1, M + 1):
    Q = [(interval(2 * m), i) for i in idx(2 * m)]
    say("products of %d entries of p: dim %d, adding the X_abcd: dim %d"
        % (m, rank(gram(Q)), rank(gram(Q + X))))

# Full flag manifold at N=3 (section 5): the elements Y_abcd = u_1a u_2b u_2c u_1d are
# invariant under the free torus, but are not in the span of the products of 2 generators
# q^s_ij = u_si u_sj of Fl^+_3 (which contains all products of at most 2 generators).
def gram_rows(elems):
    # element (rows, cols) = u_{r1 c1} ... u_{rk ck}; Haar Gram matrix h(x^* y)
    n = len(elems)
    G = np.zeros((n, n))
    k = 2 * len(elems[0][0])
    P, W = weingarten_float(k, N)
    def delta(pi, lab):
        bl = {}
        for x, v in zip(pi, lab):
            if bl.setdefault(x, v) != v:
                return 0.0
        return 1.0
    for a, (r1, c1) in enumerate(elems):
        for b, (r2, c2) in enumerate(elems):
            if b < a:
                continue
            R = tuple(r1[::-1]) + tuple(r2); C = tuple(c1[::-1]) + tuple(c2)
            dr = np.array([delta(pi, R) for pi in P]); dc = np.array([delta(pi, C) for pi in P])
            G[a, b] = G[b, a] = dr @ W @ dc
    return G
Q2 = [((s, s, t, t), c) for s in range(N) for t in range(N) for c in idx(4)]
Y = [((0, 1, 1, 0), c) for c in idx(4)]
say("full flag, N=3: products of 2 generators: dim %d, adding the Y_abcd: dim %d"
    % (rank(gram_rows(Q2)), rank(gram_rows(Q2 + Y))))
out.close()
