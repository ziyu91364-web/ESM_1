#!/usr/bin/env python3
"""Reproduce Figure 3 and independently check its analytical curves.

Requirements: Python 3, NumPy, Matplotlib. Run: python reproduce_figure3.py
All PDF/PNG figures, CSV tables and validation JSON are written beside this file.
No simulations, external files or machine-specific paths are used.
"""
from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
A = 25.
D = 2 * A + 1
GAMMA = 1.5
R = (np.sqrt(D) - 3) / 2
LM, LP = (A + 2 - np.sqrt(A * (A - 4))) / 2, (A + 2 + np.sqrt(A * (A - 4))) / 2
WM = (1.5 - LM) / (LP - LM)
WP = 1 - WM
THETA_C = 3 / np.sqrt(D)


def kernel(z):
    """(z - 1 + exp(-z))/z**2, including its continuous value at zero."""
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    small = np.abs(z) < .05
    x = z[small]
    result[small] = sum((-x) ** k / math.factorial(k + 2) for k in range(10))
    x = z[~small]
    result[~small] = (x + np.expm1(-x)) / x ** 2
    return result


def finite(time, rate):
    time = np.asarray(time, dtype=float)
    return 2 * time * (WM * kernel((LM + rate) * time) + WP * kernel((LP + rate) * time))


def variance(rate):
    return (2 * rate + 3) / (rate ** 2 + (A + 2) * rate + D)


def bisect(function, low, high):
    assert function(low) < 0 < function(high)
    for _ in range(100):
        mid = (low + high) / 2
        if function(mid) < 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def write_csv(name, headers, columns):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(zip(*columns))


def generator(a, theta):
    pi = np.array([1., a, a]) / (2 * a + 1)
    original = np.array([[-a, a, 0], [0, -1, 1], [1, 0, -1]], dtype=float)
    adjoint = original.T * pi[None, :] / pi[:, None]
    return (1 + theta) * original / 2 + (1 - theta) * adjoint / 2, pi


def centered(matrix, pi):
    root = np.sqrt(pi)
    _, _, vh = np.linalg.svd(root.reshape(1, -1), full_matrices=True)
    basis = vh[1:, :].T
    reduced = basis.T @ (-matrix * root[:, None] / root[None, :]) @ basis
    symmetric = (reduced + reduced.T) / 2
    antisymmetric = (reduced - reduced.T) / 2
    return reduced, symmetric, antisymmetric


def validate():
    # Direct covariance integration with Gauss--Legendre quadrature, separately
    # evaluated from the exact integrated kernel.
    nodes, weights = np.polynomial.legendre.leggauss(256)
    max_integral_error = 0.
    comparisons = 0
    for time in [1e-5, .001, .01, .1, .2, .5, 1., 1.368823336770668, 2., 5., 20., 100.]:
        t = (nodes + 1) * time / 2
        covariance = WM * np.exp(-LM * t) + WP * np.exp(-LP * t)
        for rate in [0., R, 7.]:
            direct = time * np.dot(weights, (1 - t / time) * np.exp(-rate * t) * covariance)
            exact = float(finite(time, rate))
            max_integral_error = max(max_integral_error, abs(direct - exact))
            assert abs(direct - exact) < 2e-10
            comparisons += 1
    # Validate full-generator covariance through an independent eigendecomposition.
    full, pi = generator(A, 1.)
    f = np.array([2 * A, -1., -1.]) / np.sqrt(2 * A)
    eigenvalues, eigenvectors = np.linalg.eig(full)
    inverse = np.linalg.inv(eigenvectors)
    max_covariance_error = 0.
    for time in [.001, .01, .1, .5, 1., 3., 10.]:
        direct = np.dot(pi * f, eigenvectors @ (np.exp(eigenvalues * time) * (inverse @ f)))
        exact = WM * np.exp(-LM * time) + WP * np.exp(-LP * time)
        max_covariance_error = max(max_covariance_error, abs(float(direct.real) - exact))
        assert abs(direct - exact) < 1e-12
    # Theta formulas checked in independent centered coordinates, using matrix
    # eigenvalues and bisection rather than the displayed threshold formulas.
    theta_tests = 0
    max_threshold_error = max_variance_error = 0.
    for a in [1., 2., 4., 5., 25., 100.]:
        for theta in sorted(set([0., .1, .2, .4, float(min(1., 3 / np.sqrt(2 * a + 1))), .5, .75, 1.])):
            full, pi = generator(a, theta)
            assert np.all(full[~np.eye(3, dtype=bool)] >= 0)
            assert np.max(np.abs(full.sum(1))) < 1e-12
            assert np.max(np.abs(pi @ full)) < 1e-12
            reduced, s, skew = centered(full, pi)
            q = lambda r: (s + r * np.eye(2)) @ (s + r * np.eye(2)) - skew.T @ skew
            mineig = lambda r: np.linalg.eigvalsh(q(r))[0]
            local = 0. if mineig(0.) >= -1e-12 else bisect(mineig, 0., 100.)
            eigs, vectors = np.linalg.eigh(s)
            si = (vectors * eigs ** -.5) @ vectors.T
            recovery = max(0., np.linalg.eigvalsh(si @ skew.T @ skew @ si - s)[-1])
            exact_local = max(0., (theta * np.sqrt(2 * a + 1) - 3) / 2)
            exact_recovery = max(0., (theta ** 2 * (2 * a + 1) - 9) / 6)
            max_threshold_error = max(max_threshold_error, abs(local - exact_local), abs(recovery - exact_recovery))
            assert abs(local - exact_local) < 1e-10
            assert abs(recovery - exact_recovery) < 1e-10
            observable = np.array([2 * a, -1., -1.]) / np.sqrt(2 * a)
            for rate in [0., .2, 1., 3., 10.]:
                reset = full + rate * (np.ones((3, 1)) * pi[None, :] - np.eye(3))
                direct = 2 * np.dot(pi * observable, np.linalg.solve(-reset + np.ones((3, 1)) * pi[None, :], observable))
                exact = (2 * rate + 3) / (rate ** 2 + (a + 2) * rate + (2 * a + 1) * (3 + theta ** 2) / 4)
                max_variance_error = max(max_variance_error, abs(direct - exact))
                assert abs(direct - exact) < 1e-11
            theta_tests += 1
    delta = variance(R) - variance(0)
    derivative_zero = (A - 4) / D ** 2
    crossing = bisect(lambda time: float(finite(time, R) - finite(time, 0)), 1., 2.)
    bracket = [1.3688233367, 1.3688233369]
    bracket_values = [float(finite(t, R) - finite(t, 0)) for t in bracket]
    assert bracket_values[0] < 0 < bracket_values[1]
    hminus, hplus = -WM / LM ** 2, WP / LP ** 2
    k = -WM / (LM + R) ** 2
    assert abs(k - WP / (LP + R) ** 2) < 1e-15
    assert hminus - hplus - k > 0
    report = dict(status="passed", a=A, reset_rate=R, theta_onset=THETA_C,
        lambda_minus=LM, lambda_plus=LP, weight_minus=WM, weight_plus=WP,
        covariance_zero=np.log(WP / (-WM)) / (LP - LM),
        crossing=crossing, crossing_bracket=bracket, crossing_bracket_differences=bracket_values,
        crossing_uniqueness="Analytical: G'' changes sign once, G' tends to a positive limit, and G tends to positive infinity.",
        spectral_relaxation_time=1 / LM, coercive_envelope_time=1 / GAMMA,
        crossing_in_spectral_times=crossing * LM, crossing_in_coercive_times=crossing * GAMMA,
        asymptotic_variance_zero=variance(0), asymptotic_variance_reset=variance(R),
        asymptotic_ratio=variance(R) / variance(0), asymptotic_difference=delta,
        derivative_at_zero=derivative_zero,
        sufficient_time_proposition_1=2 * (GAMMA ** -2 + (GAMMA + R) ** -2) / delta,
        sufficient_time_exact_coefficients=derivative_zero / delta,
        exact_coefficient_positive_margin=hminus - hplus - k,
        hminus=hminus, hplus=hplus, k=k,
        direct_covariance_integral_comparisons=comparisons,
        maximum_integral_absolute_error=max_integral_error,
        maximum_full_covariance_absolute_error=max_covariance_error,
        theta_parameter_tests=theta_tests, maximum_theta_threshold_absolute_error=max_threshold_error,
        maximum_theta_variance_absolute_error=max_variance_error)
    return report


def main():
    result = validate()
    time = np.unique(np.r_[np.geomspace(.01, 100, 1601), result["crossing"], 1 / LM, 1 / GAMMA])
    baseline, reset = finite(time, 0), finite(time, R)
    ratio = reset / baseline
    theta = np.unique(np.r_[np.linspace(0, 1, 1001), THETA_C])
    local = np.maximum(theta * np.sqrt(D) - 3, 0) / 2
    recovery = np.maximum(theta ** 2 * D - 9, 0) / 6
    write_csv("Fig3a.csv", ["observation_time", "variance_per_time_zero_reset", "variance_per_time_reset_at_local_threshold", "finite_time_ratio"], [time, baseline, reset, ratio])
    write_csv("Fig3b.csv", ["theta", "local_threshold", "recovery_threshold"], [theta, local, recovery])
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.linewidth": .65,
        "lines.linewidth": 1.45, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white"})
    blue, orange = "#17678B", "#B14B24"
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.05), layout="constrained")
    ax = axes[0]
    ax.plot(time, ratio, color=blue)
    ax.axhline(1., color=".55", linewidth=.7)
    ax.axhline(result["asymptotic_ratio"], color=orange, linestyle="--", linewidth=1.)
    ax.text(.012, result["asymptotic_ratio"] + .002, r"$T\to\infty$: 1.09179", color=orange, fontsize=8.5)
    for scale in [1 / LM, 1 / GAMMA]:
        ax.axvline(scale, color=".7", linewidth=.7, linestyle=":")
    ax.text(1 / LM * .88, 1.037, r"$\lambda_-^{-1}$", color=".4", fontsize=9, rotation=90, ha="right")
    ax.text(1 / GAMMA * 1.08, 1.037, r"$\gamma^{-1}$", color=".4", fontsize=9, rotation=90, ha="left")
    ax.plot(result["crossing"], 1., "o", color=blue, ms=4)
    ax.annotate(r"$T_\times=1.36882$", xy=(result["crossing"], 1.), xytext=(3.2, .971),
                fontsize=8.5, arrowprops=dict(arrowstyle="-", color=blue, linewidth=.7))
    ax.set(xscale="log", xlabel="Observation time $T$", ylabel=r"Finite-time ratio $V_T(r_{\rm loc})/V_T(0)$",
           xlim=(.01, 100), ylim=(.935, 1.106))
    ax.set_xticks([.01, .1, 1., 10., 100.], ["0.01", "0.1", "1", "10", "100"])
    ax = axes[1]
    ax.plot(theta, local, color=blue, label=r"$r_{\rm loc}(\theta)$")
    ax.plot(theta, recovery, color=orange, linestyle="--", label=r"$r_{\rm rec}(\theta)$")
    ax.axvline(THETA_C, color=".5", linewidth=.8, linestyle=":")
    ax.text(THETA_C + .025, 5.4, r"$\theta_c=3/\sqrt{51}$" + "\n" + r"$\simeq0.42008$", fontsize=8.5, color=".35")
    ax.plot([THETA_C], [0], "o", color=".35", ms=3.5)
    ax.set(xlabel=r"Nonreversibility parameter $\theta$", ylabel="Reset threshold", xlim=(0, 1), ylim=(-.12, 7.35))
    ax.legend(frameon=False, loc="upper left")
    for ax, letter in zip(axes, "ab"):
        ax.text(0, 1.035, f"({letter})", transform=ax.transAxes, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", width=.65, length=3)
    fig.savefig(OUT / "Fig3.pdf")
    fig.savefig(OUT / "Fig3.png", dpi=220)
    plt.close(fig)
    (OUT / "numerical_additions_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("Wrote Figure 3, both CSV tables, and validation JSON beside this script.")


if __name__ == "__main__":
    main()
