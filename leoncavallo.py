# leoncavallo.py: checks in random matrix models of the free determinantal
# varieties (section 6): the Segre map at r=1 and the partial isometry
# embedding (Theorem 6.2), and the free Hermitian determinantal varieties,
# with the Grassmannian embedding x = p/sqrt(r) and the coaction of O_N^+.
# Coordinates are matrices with entries in M_K(C), stored as arrays of shape
# (rows, cols, K, K). Results go to leoncavallo.txt.
import numpy as np

rng = np.random.default_rng(7)
out = open("leoncavallo.txt", "w")
def say(s):
    print(s); out.write(s + "\n"); out.flush()

def symunitary(K):
    # random self-adjoint unitary in M_K(C)
    Q, _ = np.linalg.qr(rng.normal(size=(K, K)) + 1j * rng.normal(size=(K, K)))
    return Q @ np.diag(rng.choice([-1.0, 1.0], size=K)) @ Q.conj().T

def orth(n):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return Q

def mul(a, b):
    # product of matrices with operator entries
    return np.einsum('ikab,kjbc->ijac', a, b)

def adj(a):
    # entrywise adjoint, with transposition: (a*)_ij = (a_ji)^*
    return a.transpose(1, 0, 3, 2).conj()

def ident(n, K):
    e = np.zeros((n, n, K, K), complex)
    for i in range(n):
        e[i, i] = np.eye(K)
    return e

def tr(a):
    return sum(a[i, i] for i in range(a.shape[0]))

def sumsq(x):
    # sum_ij x_ij x_ij^*
    return sum(x[i, j] @ x[i, j].conj().T for i in range(x.shape[0]) for j in range(x.shape[1]))

def sphere(N, K):
    # free sphere model z_i = c_i A_i, with c a unit vector and A_i self-adjoint unitaries
    c = rng.normal(size=N); c /= np.linalg.norm(c)
    return [c[i] * symunitary(K) for i in range(N)]

def grass(N, L, K):
    # p_ij = sum_r c_ri c_rj A_i A_j, a projection of trace L (as in verdi.py)
    c = orth(N)[:L]
    A = [symunitary(K) for _ in range(N)]
    p = np.zeros((N, N, K, K), complex)
    for i in range(N):
        for j in range(N):
            p[i, j] = (c[:, i] @ c[:, j]) * (A[i] @ A[j])
    return p

def report(name, errs):
    say("%-44s %s" % (name, "  ".join("%s=%.1e" % kv for kv in errs)))

# Theorem 6.2 (2): Segre map S x S -> D^+_{1MN}, with commuting legs
for (M, N, K) in [(2, 3, 2), (3, 3, 2), (3, 4, 2)]:
    say("progress: Segre model M=%d N=%d" % (M, N))
    z, w = sphere(M, K), sphere(N, K)
    I = np.eye(K)
    x = np.zeros((M, N, K * K, K * K), complex)
    for i in range(M):
        for a in range(N):
            x[i, a] = np.kron(z[i], w[a])
    p = np.zeros((M, M, K * K, K * K), complex)
    for i in range(M):
        for j in range(M):
            p[i, j] = np.kron(z[i] @ z[j], I)
    E = np.eye(K * K)
    report("Segre M=%d N=%d" % (M, N), [
        ("|x-xbar|", max(np.abs(x[i, a] - x[i, a].conj().T).max() for i in range(M) for a in range(N))),
        ("|sum x^2-1|", np.abs(sum(x[i, a] @ x[i, a] for i in range(M) for a in range(N)) - E).max()),
        ("|p^2-p|", np.abs(mul(p, p) - p).max()),
        ("|Tr p-1|", np.abs(tr(p) - E).max()),
        ("|px-x|", np.abs(mul(p, x) - x).max())])

# Theorem 6.2 (3): free partial isometries u_ij = V_ij A_j, with V a real
# partial isometry of rank r; then p = uu^t, x = u/sqrt(r)
for (M, N, r, K) in [(3, 4, 2, 3), (4, 5, 2, 3), (4, 4, 3, 2)]:
    say("progress: partial isometry model M=%d N=%d r=%d" % (M, N, r))
    V = orth(M)[:, :r] @ orth(N)[:r, :]
    A = [symunitary(K) for _ in range(N)]
    u = np.zeros((M, N, K, K), complex)
    for i in range(M):
        for j in range(N):
            u[i, j] = V[i, j] * A[j]
    ut = u.transpose(1, 0, 2, 3)
    p = mul(u, ut)
    x = u / np.sqrt(r)
    E = np.eye(K)
    report("partial isometries M=%d N=%d r=%d" % (M, N, r), [
        ("|p*-p|", np.abs(adj(p) - p).max()),
        ("|p^2-p|", np.abs(mul(p, p) - p).max()),
        ("|Tr p-r|", np.abs(tr(p) - r * E).max()),
        ("|sum x^2-1|", np.abs(sum(x[i, j] @ x[i, j] for i in range(M) for j in range(N)) - E).max()),
        ("|px-x|", np.abs(mul(p, x) - x).max()),
        ("|xx^t-p/r|", np.abs(mul(x, x.transpose(1, 0, 2, 3)) - p / r).max())])

# Hermitian determinantal varieties: x = p/sqrt(r), and the coaction of O_N^+
# in the model u_ij = O_ij g_j, with O orthogonal and g_j self-adjoint unitaries
for (N, r, K) in [(3, 1, 3), (4, 2, 2), (5, 2, 2), (5, 3, 2)]:
    say("progress: Hermitian model N=%d r=%d" % (N, r))
    p = grass(N, r, K)
    x = p / np.sqrt(r)
    E = np.eye(K)
    report("Grassmannian into D^+, r=%d N=%d" % (r, N), [
        ("|x*-x|", np.abs(adj(x) - x).max()),
        ("|sum xx*-1|", np.abs(sumsq(x) - E).max()),
        ("|px-x|", np.abs(mul(p, x) - x).max()),
        ("|xp-x|", np.abs(mul(x, p) - x).max())])
    O = orth(N)
    g = [symunitary(2) for _ in range(N)]
    u = np.zeros((N, N, 2, 2), complex)
    for i in range(N):
        for j in range(N):
            u[i, j] = O[i, j] * g[j]
    I2, E2 = np.eye(2), np.eye(2 * K)
    def coact(y):
        # Y_ij = sum_kl y_kl (x) u_ki u_lj
        Y = np.zeros((N, N, 2 * K, 2 * K), complex)
        for i in range(N):
            for j in range(N):
                Y[i, j] = sum(np.kron(y[k, l], u[k, i] @ u[l, j]) for k in range(N) for l in range(N))
        return Y
    X, P = coact(x), coact(p)
    report("O_N^+ coaction N=%d r=%d" % (N, r), [
        ("|uu^t-1|", np.abs(mul(u, u.transpose(1, 0, 2, 3)) - ident(N, 2)).max()),
        ("|X*-X|", np.abs(adj(X) - X).max()),
        ("|sum XX*-1|", np.abs(sumsq(X) - E2).max()),
        ("|P^2-P|", np.abs(mul(P, P) - P).max()),
        ("|Tr P-r|", np.abs(tr(P) - r * E2).max()),
        ("|PX-X|", np.abs(mul(P, X) - X).max())])

# at r=1, x_ij = z_i z_j from the free sphere
say("progress: r=1 sphere model")
for N in (3, 4):
    z = sphere(N, 3)
    x = np.zeros((N, N, 3, 3), complex)
    for i in range(N):
        for j in range(N):
            x[i, j] = z[i] @ z[j]
    report("sphere into D^+, r=1 N=%d" % N, [
        ("|x*-x|", np.abs(adj(x) - x).max()),
        ("|sum xx*-1|", np.abs(sumsq(x) - np.eye(3)).max()),
        ("|x^2-x|", np.abs(mul(x, x) - x).max())])
say("done")
out.close()
