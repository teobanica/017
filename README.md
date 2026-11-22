# Italian projective geometry goes free

**Claude O and Teo B**, November 2026

This repository contains the paper (32 pages, PDF and LaTeX source) and the Python programs used for its computations.

## Summary

The paper develops basic free projective geometry in the old Italian style, with all the objects given by explicit generators and relations, and with freeness as the only modern ingredient.

- **Projective spaces, Segre and Veronese.** The free projective space P^{N-1}_+ comes from the projections of trace 1. There are free analogues of the Segre and Veronese embeddings, and of the hypersurfaces as hyperplane sections. The free projective line is classical, P^1_+ = P^1_C, while C(P^{N-1}_+) is not even nuclear at N ≥ 3.
- **Grassmannians.** The free Grassmannians Gr^+_{LN} come from the projections of trace L, and embed into P^{N^2-1}_+, as a substitute for the Plücker embedding. The projective quantum isometry groups of the real, complex and free Grassmannians are PO_N ⊂ PU_N ⊂ PO_N^+ = PU_N^+, the complex case being computer-assisted, via an exact Tannakian certificate over Q(N).
- **Flag manifolds.** The free flag manifolds are the free isospectral manifolds, they come from O_N^+, with real and complex orbit models which coincide, and the coaction of O_N^+ is ergodic. The torus fixed points of the free full flag manifold are given by the quantum permutation group S_N^+.
- **Determinantal varieties.** Free determinantal varieties via incidence models, rectangular and Hermitian, related to the Segre embedding and to the free partial isometries. Free Stiefel manifolds.
- **Spectral theory.** The free Grassmannians are multiplicity free, with spectrum r_0, r_2, r_4, ... independent of L, as for the free projective spaces, and at N ≥ 3 they are not quotient spaces of O_N^+.
- **Integration and heat.** The coordinate N p_11 becomes free Poisson of parameter L, as N → ∞. The heat trace of the Brownian semigroup is finite precisely above a threshold s_N, due to the exponential growth of the multiplicities.
- **Arithmetic aspects.** The free Grassmannians are archimedean free real algebraic varieties, real forms of affine varieties defined over Q, with rational Haar integrals, and p-adic questions for the free Poisson laws.
- **Physics.** Free projective spaces as free spaces of pure states, in relation with the Standard Model, free transition probabilities, whose joint law is that of the matrix (u_st^2) over O_N^+, and a free Jarlskog invariant, whose variance is given by an explicit rational function of N, positive for N ≥ 3.

## Contents

- `017.pdf`, `017.tex`: the paper.
- The programs below, each writing its results to the `.txt` file of the same name.

| File | Content | Section |
|------|---------|---------|
| `verdi.py` | Checks of the free Segre and Veronese maps, and of the embedding of Gr^+_{LN} into P^{N^2-1}_+, in random matrix models | 3, 4 |
| `mascagni.py` | Exact Tannakian certificate over Q(N) for the projective quantum isometry group of the complex Grassmannians, with exact factorization of all denominators | 4 |
| `leoncavallo.py` | Checks of the free determinantal varieties, rectangular and Hermitian, in random matrix models, including a noncommutative model of O_N^+ | 6 |
| `donizetti.py` | Weingarten calculus for O_N^+: noncrossing pairings, Gram and Weingarten matrices (module used by the other programs) | 7 |
| `puccini.py` | Haar Gram matrix ranks showing that the Grassmannians and full flag manifolds are smaller than the corresponding quotient spaces | 7 |
| `rossini.py` | Exact moments of the coordinate p_11, and the free Poisson limit | 7 |
| `bellini.py` | Brownian eigenvalues on O_N^+ and the heat trace threshold s_N | 8 |
| `pergolesi.py` | p-adic valuations of the binomial moments of the free Poisson laws | 9 |
| `monteverdi.py` | Joint law of the free transition matrix, by brute force at small N | 10 |
| `vivaldi.py` | Exact moments of the free Jarlskog variable, and proof of the closed formula for its variance, for all N | 10 |

## Running the code

Python 3 with numpy. Run each program from the top folder, for example `python3 verdi.py`. Most programs take a few seconds, while `mascagni.py` and `vivaldi.py` take a few minutes.

## License

MIT.
