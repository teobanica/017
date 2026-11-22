# pergolesi.py: p-adic valuations of the binomial moments b_n = int binom(x,n) of the
# free Poisson law pi_t, whose moments are the Narayana polynomials (section 9).
# A law is a bounded measure on Z_p when the b_n are bounded (Mahler). Writes pergolesi.txt.
from fractions import Fraction as F
from math import comb

def moment(k, t):
    if k == 0:
        return F(1)
    return sum(F(comb(k, j) * comb(k, j - 1), k) * t ** j for j in range(1, k + 1))

def stirling1(n):
    s = [[0] * (n + 1) for _ in range(n + 1)]
    s[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            s[i][j] = s[i - 1][j - 1] - (i - 1) * s[i - 1][j]
    return s

def vp(x, p):
    if x == 0:
        return None
    a, b, v = x.numerator, x.denominator, 0
    while a % p == 0:
        a //= p; v += 1
    while b % p == 0:
        b //= p; v -= 1
    return v

NMAX = 80
S = stirling1(NMAX)
fact = [1]
for i in range(1, NMAX + 1):
    fact.append(fact[-1] * i)
out = open("pergolesi.txt", "w")
def say(s):
    print(s); out.write(s + "\n"); out.flush()
say("p  t  min_{n<=80} v_p(b_n)  v_p(b_n) at n=20,40,60,80")
for p in (2, 3, 5):
    for t in sorted({1, 2, 3, p, p ** 2, p ** 3, p ** 4}):
        m = [moment(k, t) for k in range(NMAX + 1)]
        b = [sum(S[n][j] * m[j] for j in range(n + 1)) / F(fact[n]) for n in range(NMAX + 1)]
        v = [vp(x, p) for x in b]
        say("%d  %d  %d  %s" % (p, t, min(x for x in v if x is not None), [v[n] for n in (20, 40, 60, 80)]))
# higher orders, for t = 81 at p = 3 and t = 125 at p = 5
say("first n <= 200 with v_p(b_n) < 0, for the cases where the valuations are >= 0 up to 80:")
NB = 200
S = stirling1(NB)
fact = [1]
for i in range(1, NB + 1):
    fact.append(fact[-1] * i)
for p, t in [(3, 81), (5, 125)]:
    m = [moment(k, t) for k in range(NB + 1)]
    first = None
    for n in range(NB + 1):
        b = sum(S[n][j] * m[j] for j in range(n + 1)) / F(fact[n])
        v = vp(b, p)
        if v is not None and v < 0:
            first = (n, v); break
    say("%d  %d  first negative valuation at n=%s, v_p(b_n)=%s" % (p, t, first[0], first[1]))
out.close()
