#!/usr/bin/env python3
"""Validate Appendix A using full expanded generators.

Requires Python 3, NumPy and SciPy. Run: python validate_splitting.py
Writes validate_splitting_results.json beside this script. No external data.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag, eigh, expm, null_space

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261021)
ERRORS = {}


def check(name, actual, expected, tolerance=2e-9):
    err = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    ERRORS[name] = max(ERRORS.get(name, 0.), err)
    if err > tolerance:
        raise AssertionError(f"{name}: {err} > {tolerance}")


def base_chain(a, theta):
    p, m = (1 + theta) / 2, (1 - theta) / 2
    L = np.array([[-a, a*p, a*m], [m, -1., p], [p, m, -1.]])
    pi = np.array([1., a, a]) / (2*a + 1)
    f = np.array([2*a, -1., -1.]) / np.sqrt(2*a)
    return L, pi, f


def split(L, pi, sizes):
    groups = np.repeat(np.arange(len(pi)), sizes)
    rho = [RNG.dirichlet(np.ones(m) * 2) for m in sizes]
    weights = np.concatenate(rho)
    kappa = RNG.uniform(.2, 3., len(pi))
    n = len(groups)
    Q = np.zeros((n, n))
    for u, i in enumerate(groups):
        for v, j in enumerate(groups):
            if u != v:
                Q[u, v] = (L[i, j] if i != j else kappa[i]) * weights[v]
    Q -= np.diag(Q.sum(axis=1))
    pit = pi[groups] * weights
    U = np.zeros((n, len(pi)))
    internal = []
    offset = 0
    for i, (m, ri) in enumerate(zip(sizes, rho)):
        U[offset:offset+m, i] = np.sqrt(ri)
        if m > 1:
            W = np.zeros((n, m-1))
            W[offset:offset+m] = null_space(np.sqrt(ri)[None, :])
            internal.append(W)
        offset += m
    O = np.column_stack([U] + internal)
    extra = np.repeat(-np.diag(L) + kappa, np.array(sizes)-1)
    return Q, pit, groups, O, extra


def centered(L, pi):
    root = np.sqrt(pi)
    B = null_space(root[None, :])
    full = -root[:, None] * L / root[None, :]
    M = B.T @ full @ B
    return M, (M+M.T)/2, (M-M.T)/2, B


def thresholds(S, A):
    I = np.eye(len(S))
    AA = A.T @ A
    def smallest(r):
        B = S+r*I
        return np.linalg.eigvalsh(B@B-AA)[0]
    if smallest(0.) >= -1e-11:
        loc = 0.
    else:
        lo, hi = 0., max(1., np.linalg.norm(A, 2))
        while smallest(hi) < 0:
            hi *= 2
        for _ in range(70):
            mid = (lo+hi)/2
            if smallest(mid) < 0:
                lo = mid
            else:
                hi = mid
        loc = (lo+hi)/2
    # Independent generalized-eigenvalue form of baseline recovery.
    rec = max(0., float(eigh(AA-S@S, S, eigvals_only=True)[-1]))
    return np.array([loc, rec])


def finite_variance(M, B, pi, f, r, T):
    coeff = B.T @ (np.sqrt(pi)*f)
    Mr = M+r*np.eye(len(M))
    R = np.linalg.inv(Mr)
    K = R-R@R@(np.eye(len(M))-expm(-T*Mr))/T
    return float(2*coeff@K@coeff)


cases = []
for a, theta in [(3., .6), (25., .25), (25., .8), (100., .9)]:
    L, pi, f = base_chain(a, theta)
    M, S, A, B = centered(L, pi)
    expected = np.array([max(0., (theta*np.sqrt(2*a+1)-3)/2),
                         max(0., (theta*theta*(2*a+1)-9)/6)])
    check("base_threshold_formula", thresholds(S, A), expected)
    for sizes in [(1, 1, 1), (2, 1, 1), (1, 3, 1), (2, 3, 4), (4, 5, 7)]:
        Q, pit, groups, O, extra = split(L, pi, sizes)
        ft = f[groups]
        root = np.sqrt(pit)
        full = -root[:, None]*Q/root[None, :]
        basefull = -np.sqrt(pi)[:, None]*L/np.sqrt(pi)[None, :]
        expected_block = block_diag(basefull, np.diag(extra)) if len(extra) else basefull
        check("row_sums", Q.sum(axis=1), 0.)
        check("stationarity", pit@Q, 0.)
        check("normalization", pit.sum(), 1.)
        check("orthogonal_decomposition", O.T@O, np.eye(len(pit)))
        check("full_generator_blocks", O.T@full@O, expected_block)
        check("full_adjoint_blocks", O.T@full.T@O, expected_block.T)
        check("lift_norm", pit@(ft*ft), pi@(f*f))
        check("lift_centering", pit@ft, 0.)
        if np.min(Q[~np.eye(len(Q), dtype=bool)]) <= 0:
            raise AssertionError("Expected every off-diagonal rate positive")
        Mt, St, At, Bt = centered(Q, pit)
        observed = thresholds(St, At)
        check("expanded_thresholds", observed, expected)
        for r in [0., .8]:
            Lr = L+r*(np.ones((3, 1))*pi-np.eye(3))
            Qr = Q+r*(np.ones((len(pit), 1))*pit-np.eye(len(pit)))
            for t in [0., .05, .7, 3.]:
                cb = float((pi*f)@expm(t*Lr)@f)
                ct = float((pit*ft)@expm(t*Qr)@ft)
                check("full_covariance_lift", ct, cb)
            for T in [.1, .7, 3.]:
                vb = finite_variance(M, B, pi, f, r, T)
                vt = finite_variance(Mt, Bt, pit, ft, r, T)
                check("finite_time_variance_lift", vt, vb)
        if a > 4:
            check("real_relaxation_spectrum", np.linalg.eigvals(-Q).imag, 0.)
        cases.append({"a": a, "theta": theta, "group_sizes": sizes,
                      "state_count": len(pit), "local_threshold": float(observed[0]),
                      "recovery_threshold": float(observed[1])})

report = {"status": "passed", "seed": 20261021, "case_count": len(cases),
          "state_counts": sorted(set(c["state_count"] for c in cases)),
          "maximum_absolute_errors": ERRORS, "cases": cases,
          "scope": "Full generators, full weighted adjoints, exact extra reversible blocks, stationary law, both universal thresholds, lifted reset covariance, and lifted finite-time variance."}
destination = OUT / "validate_splitting_results.json"
destination.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps({"status": report["status"], "cases": len(cases),
                  "max_error": max(ERRORS.values()), "report": destination.name}))
