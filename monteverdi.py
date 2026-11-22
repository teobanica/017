# monteverdi.py: the free transition variable T = sum_ij q_ij (x) r_ij between
# 2 independent points q in Gr^+_{L,N}, r in Gr^+_{K,N} (section 10).
# (1) exact moments of T, by brute force over the indices, versus the moments of
#     the block variable sum_{a<=L, b<=K} u_ab^2 over O_N^+;
# (2) the same for the transposed variable T' = sum_ij q_ij (x) r_ji, whose law differs.
# Results go to monteverdi.txt.
import itertools
from fractions import Fraction as F
from donizetti import nc2, join_blocks, interval, solve_exact

out = open("monteverdi.txt", "w")
def say(s):
    print(s); out.write(s + "\n"); out.flush()

def wmatrix(n, N):
    P = nc2(n)
    G = [[F(N) ** join_blocks(a, b) for b in P] for a in P]
    W = []
    for j in range(len(P)):
        e = [F(int(i == j)) for i in range(len(P))]
        W.append(solve_exact(G, e))
    return P, W

def delta(pi, word):
    blocks = {}
    for b, x in zip(pi, word):
        if blocks.setdefault(b, x) != x:
            return 0
    return 1

def gr_integral(word, L, P, W):
    # integral over Gr^+_{L,N} of p_{w1w2} p_{w3w4} ..., word = (w1,w2,...)
    tau = interval(len(word))
    v = [F(L) ** join_blocks(pi, tau) for pi in P]
    d = [delta(s, word) for s in P]
    return sum(v[a] * W[a][b] * d[b] for a in range(len(P)) for b in range(len(P)) if d[b])

def block_moment(k, N, L, K, P, W):
    tau = interval(2 * k)
    vL = [F(L) ** join_blocks(pi, tau) for pi in P]
    vK = [F(K) ** join_blocks(pi, tau) for pi in P]
    return sum(vL[a] * W[a][b] * vK[b] for a in range(len(P)) for b in range(len(P)))

for (N, L, K, kmax) in [(3, 1, 1, 3), (3, 1, 2, 3), (3, 2, 2, 3), (4, 2, 1, 2), (4, 2, 2, 2)]:
    say("N=%d L=%d K=%d" % (N, L, K))
    for k in range(1, kmax + 1):
        P, W = wmatrix(2 * k, N)
        cache = {}
        def gi(word, M):
            key = (word, M)
            if key not in cache:
                cache[key] = gr_integral(word, M, P, W)
            return cache[key]
        T = F(0); Tt = F(0)
        for I in itertools.product(range(N), repeat=2 * k):
            a = gi(I, L)
            if a == 0:
                continue
            Isw = tuple(I[2 * l + 1 - t] for l in range(k) for t in range(2))
            T += a * gi(I, K)
            Tt += a * gi(Isw, K)
        B = block_moment(k, N, L, K, P, W)
        say("  k=%d  E[T^k]=%s  block=%s  equal=%s  E[T'^k]=%s" % (k, T, B, T == B, Tt))

out.close()
