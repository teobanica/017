# vivaldi.py: the free Jarlskog variable J = (X - X^*)/2i, where
# X = sum_{ijkl} q^1_ij q^2_kl (x) r^1_jk r^2_li, for 2 independent free full flags q, r
# in Fl^+_N, with the integration coming from O_N^+ (section 10). Exact moments via the
# Weingarten formula over NC_2, with the 2 legs contracted as in Theorem 10.5.
# As a check, the same computation with all pairings (classical O_N, N=4) gives J = 0.
# Also proves the closed formula for int J^2, valid for all N, by exact evaluation at 129 points.
# Writes vivaldi.txt.
from fractions import Fraction as F
from donizetti import nc2, join_blocks, solve_exact

def all_pairings(k):
    if k == 0:
        return [()]
    res = []
    def rec(lab, nxt):
        if None not in lab:
            res.append(tuple(lab)); return
        a = lab.index(None)
        for b in range(a + 1, k):
            if lab[b] is None:
                l2 = list(lab); l2[a] = l2[b] = nxt
                rec(l2, nxt + 1)
    rec([None] * k, 0)
    return res

CACHE = {}
def weingarten(n, N, pairings):
    key = (n, N, pairings.__name__)
    if key not in CACHE:
        P = pairings(n)
        G = [[F(N) ** join_blocks(a, b) for b in P] for a in P]
        W = [solve_exact(G, [F(int(i == j)) for i in range(len(P))]) for j in range(len(P))]
        CACHE[key] = (P, W)
    return CACHE[key]

X = ([(1, 'i', 'j'), (2, 'k', 'l')], [(1, 'j', 'k'), (2, 'l', 'i')])
Xs = ([(2, 'l', 'k'), (1, 'j', 'i')], [(2, 'i', 'l'), (1, 'k', 'j')])

def expectation(factors, N, pairings):
    q, r = [], []
    for t, (qq, rr) in enumerate(factors):
        q += [(s, a + str(t), b + str(t)) for s, a, b in qq]
        r += [(s, a + str(t), b + str(t)) for s, a, b in rr]
    P, W = weingarten(2 * len(q), N, pairings)
    def leg(word):
        alpha, gam = [], []
        for s, a, b in word:
            alpha += [s, s]; gam += [a, b]
        def d(pi):
            bl = {}
            for x, v in zip(pi, alpha):
                if bl.setdefault(x, v) != v:
                    return 0
            return 1
        dp = [d(pi) for pi in P]
        return [sum(dp[a] * W[a][b] for a in range(len(P))) for b in range(len(P))], gam
    wq, gq = leg(q)
    wr, gr = leg(r)
    names = sorted(set(gq))
    def comps(sq, sr):
        par = {v: v for v in names}
        def f(x):
            while par[x] != x:
                x = par[x]
            return x
        for sig, g in ((sq, gq), (sr, gr)):
            first = {}
            for b, v in zip(sig, g):
                if b in first:
                    par[f(v)] = f(first[b])
                else:
                    first[b] = v
        return len({f(v) for v in names})
    tot = F(0)
    for a, sq in enumerate(P):
        if wq[a]:
            for b, sr in enumerate(P):
                if wr[b]:
                    tot += wq[a] * wr[b] * F(N) ** comps(sq, sr)
    return tot

out = open("vivaldi.txt", "w")
def say(s):
    print(s); out.write(s + "\n"); out.flush()

def jarlskog(N, pairings):
    EX = expectation([X], N, pairings)
    a = expectation([X, X], N, pairings); b = expectation([X, Xs], N, pairings)
    c = expectation([Xs, X], N, pairings); d = expectation([Xs, Xs], N, pairings)
    return EX, -(a - b - c + d) / 4

for label, N, pairings in [("free", 3, nc2), ("free", 4, nc2), ("free", 5, nc2), ("classical", 4, all_pairings)]:
    EX, J2 = jarlskog(N, pairings)
    say("%s N=%d" % (label, N))
    say("  int X = %s   int J^2 = %s" % (EX, J2))

# All N: the closed formulas below are checked exactly at N = 2,...,130. The exact value
# of int J^2 is a rational function of N, of the form A(N)/det(G_8)^2, with G_8 the 14 x 14
# Gram matrix of NC_2(8), whose entries are N^c with c <= 4, and with deg A <= 52+52+8 = 112.
# Thus the difference with the formula, after clearing denominators, is a polynomial of
# degree <= 127, and its vanishing at 129 points proves the formula for all N >= 2.
def P(N):
    return (N**10 - 9*N**8 - 2*N**7 + 32*N**6 + 8*N**5 - 46*N**4 - 18*N**3 + 29*N**2 + 14*N - 4)
def formula(N):
    N = F(N)
    return (N - 2) * P(N) / (2 * N * (N**2 - 1) * (N**2 - 2)**2 * (N**4 - 3*N**2 + 1)**2)
say("all N: int X = -1/N(N^2-1), int J^2 = (N-2)P(N)/2N(N^2-1)(N^2-2)^2(N^4-3N^2+1)^2, with")
say("  P(N) = N^10-9N^8-2N^7+32N^6+8N^5-46N^4-18N^3+29N^2+14N-4, checked at N = 2,...,130:")
for N in range(2, 131):
    EX, J2 = jarlskog(N, nc2)
    assert EX == F(-1, N * (N**2 - 1)) and J2 == formula(N), N
    if N <= 6 or N % 20 == 0:
        say("  N=%d  int J^2 = %s  ok" % (N, J2))
# positivity for N >= 3: P(t+3) has positive coefficients
from math import comb
c = [0] * 11
for k, a in [(10, 1), (8, -9), (7, -2), (6, 32), (5, 8), (4, -46), (3, -18), (2, 29), (1, 14), (0, -4)]:
    for j in range(k + 1):
        c[j] += a * comb(k, j) * 3 ** (k - j)
assert all(x > 0 for x in c)
say("  P(t+3) has coefficients %s, all positive, so int J^2 > 0 for N >= 3" % c)
out.close()
