# bellini.py: Brownian eigenvalues lambda_j = N U_j'(N)/U_j(N) on O_N^+, normalized
# by lambda_1=1, and the heat trace threshold s_N for the free Grassmannians
# (section 8), with heat semigroup e^{-sL}, L being the positive Laplacian.
# Writes bellini.txt.
from math import sqrt, log, exp

def cheb(N, J):
    U, dU = [1.0, N], [0.0, 1.0]
    for j in range(1, J):
        U.append(N*U[j]-U[j-1]); dU.append(U[j]+N*dU[j]-dU[j-1])
    return U, dU

out = []
for N in [3, 4, 5, 10]:
    U, dU = cheb(N, 400)
    c = [N*dU[j]/U[j] for j in range(len(U))]
    g = (N+sqrt(N*N-4))/2
    tN = sqrt(N*N-4)*log(g)/N
    out.append("N=%d  lambda_1=%.6f  lambda_2=%.6f (2N^2/(N^2-1)=%.6f)" % (N, c[1], c[2], 2*N*N/(N*N-1)))
    out.append("  slope lambda_j/j at j=300: %.6f, N/sqrt(N^2-4)=%.6f" % (c[300]/300, N/sqrt(N*N-4)))
    out.append("  s_N=%.6f" % tN)
    for f in [0.99, 1.01]:
        t = f*tN
        terms = [log(U[2*k])-t*c[2*k] for k in (50, 100, 150)]
        out.append("  s=%.2f s_N: log of terms k=50,100,150: %s" % (f, ", ".join("%.4f" % x for x in terms)))
open("bellini.txt", "w").write("\n".join(out)+"\n")
print("\n".join(out))
