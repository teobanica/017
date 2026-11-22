# Exact moments of the coordinate p_11 over the free Grassmannian Gr^+_{L,N},
# via the Weingarten formula for O_N^+ (section 7), and the N -> infinity
# check of the free Poisson limit (Theorem 7.5). Results go to rossini.txt.
import sys
from fractions import Fraction as F
from donizetti import nc2, join_blocks, interval, solve_exact, weingarten_float
import numpy as np

def moment(k, N, L):
    P = nc2(2 * k)
    G = [[F(N) ** join_blocks(a, b) for b in P] for a in P]
    w = solve_exact(G, [F(1)] * len(P))      # row sums of the Weingarten matrix
    tau = interval(2 * k)
    return sum(F(L) ** join_blocks(pi, tau) * wi for pi, wi in zip(P, w))

def free_poisson(k, L):
    P = nc2(2 * k)
    tau = interval(2 * k)
    return sum(L ** join_blocks(pi, tau) for pi in P)

out = open("rossini.txt", "w")
def say(s):
    print(s); out.write(s + "\n"); out.flush()

say("Exact moments of p_11 over Gr^+_{L,N}")
for N in range(2, 7):
    for L in range(1, N):
        say("N=%d L=%d : %s" % (N, L, ", ".join(str(moment(k, N, L)) for k in (1, 2, 3, 4))))
say("Moments of N p_11, large N, versus free Poisson pi_L")
for L in (1, 2, 3):
    for N in (10, 100, 1000):
        vals = []
        for k in (1, 2, 3):
            P, W = weingarten_float(2 * k, N)
            tau = interval(2 * k)
            v = np.array([float(L) ** join_blocks(pi, tau) for pi in P])
            vals.append(N ** k * v @ W.sum(axis=1))
        say("L=%d N=%d : %s" % (L, N, ", ".join("%.4f" % x for x in vals)))
    say("L=%d free Poisson : %s" % (L, ", ".join(str(free_poisson(k, L)) for k in (1, 2, 3))))
out.close()
