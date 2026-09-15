Online Resource 1: Analytical figures and numerical verification

Article: Variance ordering under stationary resetting: exact thresholds
and a three-state counterexample
Intended journal: Journal of Statistical Physics
Human authors, affiliations and corresponding contact: to be supplied
before submission, consistently with the manuscript title page.

Contents:
  validate.py                 NumPy-only numerical audit
  validation_results.json     Actual output of the supplied validator
  reproduce_figures.py         Analytical figure and CSV generation
  requirements.txt             Python dependencies
  Fig1a.csv, Fig1b.csv          Underlying analytical values for Figure 1
  Fig2a.csv, Fig2b.csv          Underlying analytical values for Figure 2

Run using Python 3.10 or later:
  python -m pip install -r requirements.txt
  python validate.py
  python reproduce_figures.py

The scripts write files beside themselves. The validator uses fixed seed
702319 and provides command-line options through --help. All plots are
analytical evaluations of displayed equations; no empirical data were
collected and no simulated measurements are presented as observations.

The validator tests covariance congruences, inertia, local and recovery
thresholds, the three-state formulas and isospectral comparator, normal
ring Fourier formulas, and the finite-time bound. Floating-point checks
support consistency; the manuscript's analytical proofs establish the
general results. Small numerical residual differences across platforms
are expected.

Substantive AI assistance, including code generation and separate agent
checks, is disclosed in the manuscript. Human scientific verification and
authorship remain required before journal submission.
