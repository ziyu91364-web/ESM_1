#!/usr/bin/env python3
"""Validate the nonnegative-theta extension of the three-state operator formulas.

Requires Python 3 and NumPy only. Run:
    python validate_theta_extension.py

The report is written to validate_theta_extension_results.json beside this file.
No other script, input data, or machine-specific path is required.

IMPORTANT DOMAIN DISTINCTION
For 0 <= theta <= 1, L_theta is a Markov generator; its six off-diagonal
rates are strictly positive only when theta < 1. For theta > 1, three rates
are negative. Tests in that range concern the formal accretive operator
M_theta = S + theta*A and its resolvent quadratic form, not a Markov process
or a stochastic variance. Sampled numerical checks supplement, not replace,
the analytical statements for every nonnegative theta.
"""

import json
from pathlib import Path

import numpy as np


ERRORS = {}


def check(name, actual, expected, tolerance=2e-9):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    absolute = float(np.max(np.abs(actual - expected)))
    scaled = absolute / max(1., float(np.max(np.abs(expected))))
    previous = ERRORS.setdefault(name, {"maximum_absolute_error": 0., "maximum_scaled_error": 0.})
    previous["maximum_absolute_error"] = max(previous["maximum_absolute_error"], absolute)
    previous["maximum_scaled_error"] = max(previous["maximum_scaled_error"], scaled)
    if scaled > tolerance:
        raise AssertionError(f"{name}: scaled error {scaled} exceeds {tolerance}")


def symmetric(matrix):
    return (matrix + matrix.T) / 2


def positive_boundary(smallest_eigenvalue, scale):
    """Numerical terminal PSD boundary, without using the closed-form threshold."""
    if smallest_eigenvalue(0.) >= -2e-12 * max(1., scale):
        return 0.
    low, high = 0., 1.
    while smallest_eigenvalue(high) < 0:
        high *= 2
    for _ in range(80):
        midpoint = (low + high) / 2
        if smallest_eigenvalue(midpoint) < 0:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def rates_and_basis(a, theta):
    """The algebraic L_theta matrix, including signed rates when theta > 1."""
    forward, backward = (1 + theta) / 2, (1 - theta) / 2
    matrix = np.array([
        [-a, a * forward, a * backward],
        [backward, -1., forward],
        [forward, backward, -1.]])
    pi = np.array([1., a, a]) / (2 * a + 1)
    e1 = np.array([2 * a, -1., -1.]) / np.sqrt(2 * a)
    e2 = np.array([0., 1., -1.]) * np.sqrt((2 * a + 1) / (2 * a))
    return matrix, pi, np.column_stack([e1, e2])


def run_case(a, theta):
    d = 2 * a + 1
    c = theta * np.sqrt(d) / 2
    original_m = np.array([[a + .5, -c], [c, 1.5]])
    full, pi, basis = rates_and_basis(a, theta)
    check("row_sums", full.sum(axis=1), np.zeros(3))
    check("stationary_vector_identity", pi @ full, np.zeros(3))
    check("basis_orthonormality", basis.T @ np.diag(pi) @ basis, np.eye(2))
    check("basis_centering", pi @ basis, np.zeros(2))
    check("full_matrix_restriction", basis.T @ np.diag(pi) @ (-full) @ basis, original_m)
    offdiagonal = full[~np.eye(3, dtype=bool)]
    if theta < 1:
        assert np.all(offdiagonal > 0), "All six rates must be positive for theta < 1."
        domain = "Markov generator, all six off-diagonal rates strictly positive"
    elif theta == 1:
        assert np.sum(offdiagonal > 0) == 3 and np.sum(offdiagonal == 0) == 3
        domain = "Markov generator, directed-cycle endpoint"
    else:
        assert np.sum(offdiagonal < 0) == 3 and np.sum(offdiagonal > 0) == 3
        domain = "Formal operator only; not a Markov generator (three negative rates)"

    # Rotate the named basis before taking numerical eigenvalues. The checks
    # below thus do not simply read the diagonal entries used in the proof.
    angle = .371 + .013 * np.log1p(a + theta)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    matrix = rotation.T @ original_m @ rotation
    observable = rotation.T @ np.array([1., 0.])
    s = symmetric(matrix)
    skew = (matrix - matrix.T) / 2
    aa = skew.T @ skew
    identity = np.eye(2)
    assert np.linalg.eigvalsh(s)[0] > 0

    qlocal = lambda r: (s + r * identity) @ (s + r * identity) - aa
    qbaseline = lambda r: s @ s + r * s - aa
    local = positive_boundary(lambda r: np.linalg.eigvalsh(qlocal(r))[0], np.linalg.norm(aa, 2))
    recovery_root = positive_boundary(lambda r: np.linalg.eigvalsh(qbaseline(r))[0], np.linalg.norm(aa, 2))
    eigenvalues, eigenvectors = np.linalg.eigh(s)
    inverse_root = (eigenvectors * eigenvalues ** -.5) @ eigenvectors.T
    recovery_eigenvalue = max(0., float(np.linalg.eigvalsh(inverse_root @ aa @ inverse_root - s)[-1]))
    expected_local = max(0., (theta * np.sqrt(d) - 3) / 2)
    expected_recovery = max(0., (theta ** 2 * d - 9) / 6)
    check("local_matrix_root", local, expected_local)
    check("recovery_matrix_root", recovery_root, expected_recovery)
    check("recovery_generalized_eigenvalue", recovery_eigenvalue, expected_recovery)

    denominator = lambda r: (r + a + .5) * (r + 1.5) + theta ** 2 * d / 4
    resolvent_zero = np.linalg.inv(matrix)
    response_zero = float(2 * observable @ resolvent_zero @ observable)
    rvalues = sorted(set([0., .01, .2, 1., 3., 10., local, recovery_root,
                         .75 * local, 1.25 * local, .75 * recovery_root, 1.25 * recovery_root]))
    for r in rvalues:
        inverse = np.linalg.inv(matrix + r * identity)
        direct = float(2 * observable @ inverse @ observable)
        derivative = float(-2 * observable @ inverse @ inverse @ observable)
        exact = (2 * r + 3) / denominator(r)
        exact_derivative = ((theta ** 2 * d - 9) / 2 - 6 * r - 2 * r ** 2) / denominator(r) ** 2
        exact_baseline_difference = r * ((theta ** 2 * d - 9) / 2 - 3 * r) / (denominator(0.) * denominator(r))
        check("e1_formal_quadratic_response", direct, exact)
        check("e1_derivative_identity", derivative, exact_derivative)
        check("e1_baseline_difference_identity", direct - response_zero, exact_baseline_difference)
        assert direct > 0
        # Direct congruences check the whole two-dimensional quadratic form.
        inverse_squared = inverse @ inverse
        check("derivative_quadratic_form", symmetric(inverse_squared), inverse_squared.T @ qlocal(r) @ inverse_squared)
        if r > 0:
            product_inverse = np.linalg.inv(matrix @ (matrix + r * identity))
            check("baseline_quadratic_form", symmetric((resolvent_zero - inverse) / r),
                  product_inverse.T @ qbaseline(r) @ product_inverse)

    if expected_local > 1e-8:
        for factor in [.75, 1.25]:
            inverse = np.linalg.inv(matrix + factor * expected_local * identity)
            derivative = float(-2 * observable @ inverse @ inverse @ observable)
            assert (derivative > 0) == (factor < 1), "Incorrect e1 derivative sign at local threshold."
    if expected_recovery > 1e-8:
        for factor in [.75, 1.25]:
            inverse = np.linalg.inv(matrix + factor * expected_recovery * identity)
            difference = float(2 * observable @ (inverse - resolvent_zero) @ observable)
            assert (difference > 0) == (factor < 1), "Incorrect e1 baseline sign at recovery threshold."

    return {"a": a, "theta": theta, "domain": domain,
            "minimum_off_diagonal_entry": float(np.min(offdiagonal)),
            "local_threshold": local, "recovery_threshold": recovery_root,
            "response_rates_checked": len(rvalues)}


def main():
    cases = []
    for a in [1., 2., 4., 5., 25., 100.]:
        onset = 3 / np.sqrt(2 * a + 1)
        theta_values = sorted(set([0., .1, .5, 1., 1.0001, 1.25, 2., 5., 20.,
                                   .999 * onset, onset, 1.001 * onset]))
        for theta in theta_values:
            cases.append(run_case(a, theta))
    markov_cases = int(sum(case["theta"] <= 1 for case in cases))
    report = {
        "status": "passed", "numpy_version": np.__version__,
        "scope": "Sampled nonnegative-theta checks of formal accretive-operator identities; stochastic interpretation only for 0 <= theta <= 1.",
        "a_values": [1., 2., 4., 5., 25., 100.],
        "case_count": len(cases), "markov_parameter_cases": markov_cases,
        "formal_non_markov_parameter_cases": len(cases) - markov_cases,
        "response_rate_comparisons": sum(case["response_rates_checked"] for case in cases),
        "rate_domain_result": "All six off-diagonal rates positive for sampled theta < 1; three zero at theta = 1; exactly three negative at every sampled theta > 1.",
        "maximum_errors": ERRORS, "cases": cases,
        "interpretation": "The all-theta formulas are analytical identities. Finite numerical checks are consistency tests, not a proof over the continuum or a Markov interpretation beyond theta = 1."
    }
    destination = Path(__file__).resolve().with_name("validate_theta_extension_results.json")
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "case_count": len(cases),
                      "markov_parameter_cases": markov_cases,
                      "formal_non_markov_parameter_cases": len(cases) - markov_cases,
                      "response_rate_comparisons": report["response_rate_comparisons"],
                      "maximum_scaled_error": max(item["maximum_scaled_error"] for item in ERRORS.values()),
                      "report": destination.name}, indent=2))


if __name__ == "__main__":
    main()
