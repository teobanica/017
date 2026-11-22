# Checks in random matrix models of the free Veronese maps, of the free
# Segre maps, and of the embedding Gr^+_{L,N} into P^{N^2-1}_+ (Theorems 2.4,
# 3.2 and 4.4). A model of Gr^+_{L,N} is a projection P in M_N(M_K(C)) whose
# blocks satisfy sum_i P_ii = L 1_K, built from a free Stiefel model. Results go to verdi.txt.
import itertools
import numpy as np

rng = np.random.default_rng(1)
out = open("verdi.txt", "w")
def say(s):
    print(s); out.write(s + "\n"); out.flush()

def grass_model(N, L, K):
    # u_ri = c_ri A_i, with c a real L x N matrix with orthonormal rows, and
    # A_i random self-adjoint unitaries in M_K(C); then u = u-bar, uu^t = 1_L,
    # and p_ij = sum_r u_ri u_rj is a projection of trace L, with
    # noncommuting entries
    c, _ = np.linalg.qr(rng.normal(size=(N, N)))
    c = c[:L]
    A = []
    for i in range(N):
        Z = rng.normal(size=(K, K)) + 1j * rng.normal(size=(K, K))
        Q, _ = np.linalg.qr(Z)
        D = np.diag(rng.choice([-1.0, 1.0], size=K))
        A.append(Q @ D @ Q.conj().T)
    p = np.zeros((N, N, K, K), complex)
    for i in range(N):
        for j in range(N):
            p[i, j] = sum(c[r, i] * c[r, j] for r in range(L)) * (A[i] @ A[j])
    return p

def as_matrix(p):
    N, _, K, _ = p.shape
    return p.transpose(0, 2, 1, 3).reshape(N * K, N * K)

def check(P, name, trace):
    M = as_matrix(P)
    N, K = P.shape[0], P.shape[2]
    err1 = np.abs(M - M.conj().T).max()
    err2 = np.abs(M @ M - M).max()
    err3 = np.abs(sum(P[i, i] for i in range(N)) - trace * np.eye(K)).max()
    say("%s: |P-P*| = %.1e, |P^2-P| = %.1e, |Tr P - %d| = %.1e" % (name, err1, err2, trace, err3))

N, K = 3, 4
p = grass_model(N, 1, K)
check(p, "P^2_+ model", 1)
for d in (2, 3):
    I = list(itertools.product(range(N), repeat=d))
    P = np.zeros((len(I), len(I), K, K), complex)
    for a, Ia in enumerate(I):
        for b, Jb in enumerate(I):
            w = list(Ia) + list(Jb[::-1])
            M = np.eye(K)
            for t in range(d):
                M = M @ p[w[2 * t], w[2 * t + 1]]
            P[a, b] = M
    check(P, "Veronese d=%d" % d, 1)

for L in (1, 2):
    q = grass_model(4, L, K)
    I = list(itertools.product(range(4), repeat=2))
    P = np.zeros((16, 16, K, K), complex)
    for a, (i1, i2) in enumerate(I):
        for b, (j1, j2) in enumerate(I):
            P[a, b] = q[i1, i2] @ q[j2, j1] / L
    check(P, "Gr^+_{%d,4} into P^15_+" % L, 1)

# Segre: p in Gr_{1,2} and q in Gr_{2,3}, commuting legs via Kronecker products
p = grass_model(2, 1, 2); q = grass_model(3, 2, 2)
P = np.zeros((6, 6, 4, 4), complex)
for i, a, j, b in itertools.product(range(2), range(3), range(2), range(3)):
    P[3 * i + a, 3 * j + b] = np.kron(p[i, j], q[a, b])
check(P, "Segre Gr_{1,2} x Gr_{2,3}", 2)
out.close()
