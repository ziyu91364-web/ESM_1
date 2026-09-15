#!/usr/bin/env python3
"""Reproduce analytical figures and underlying CSV tables for manuscript.tex.

Run in a Python environment with numpy and matplotlib:
    python reproduce_figures.py
Outputs are written next to this script. No simulated or empirical data are used.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11, "axes.labelsize": 11, "legend.fontsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.linewidth": 0.65, "lines.linewidth": 1.35,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
COLORS = ["#17678B", "#B14B24", "#222222"]

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.eps")
    plt.close(fig)

def table(name, header, columns):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(zip(*columns))

def panel(ax, letter):
    ax.text(0, 1.035, f"({letter})", transform=ax.transAxes, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.65)

def variance(a, r):
    return (2*r+3)/(r*r+(a+2)*r+2*a+1)

fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.0), layout="constrained")
r = np.linspace(0, 12, 1201)
cols = [r]
for a, color, style in zip([3, 25, 100], COLORS, [":", "--", "-"]):
    ratio = variance(a, r) / variance(a, 0)
    axes[0].plot(r, ratio, color=color, linestyle=style, label=f"a = {a}")
    cols.append(ratio)
    if a > 4:
        peak = (np.sqrt(2*a+1)-3)/2
        axes[0].plot(peak, variance(a, peak)/variance(a, 0), "o", color=color, ms=3.5)
axes[0].axhline(1, color="0.55", linewidth=0.65)
axes[0].set(xlabel="Reset rate r", ylabel="Relative variance  v(r) / v(0)", xlim=(0, 12), ylim=(0, 1.24))
axes[0].legend(frameon=False, loc="lower left")
a = np.linspace(1, 100, 991)
local = np.maximum(np.sqrt(2*a+1)-3, 0)/2
recovery = np.maximum(a-4, 0)/3
axes[1].plot(a, local, color=COLORS[0], label=r"$r_{\mathrm{loc}}$", linestyle="-")
axes[1].plot(a, recovery, color=COLORS[1], label=r"$r_{\mathrm{rec}}$", linestyle="--")
axes[1].set(xlabel="Fast transition rate a", ylabel="Reset threshold", xlim=(1, 100), ylim=(0, 34))
axes[1].legend(frameon=False, loc="upper left")
for ax, letter in zip(axes, "ab"):
    panel(ax, letter)
save(fig, "Fig1")
table("Fig1a.csv", ["reset_rate", "relative_variance_a3", "relative_variance_a25", "relative_variance_a100"], cols)
table("Fig1b.csv", ["a", "local_threshold", "recovery_threshold"], [a, local, recovery])

fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.0), layout="constrained")
a = 25.0
t = np.linspace(0.15, 2.0, 1201)
delta = np.sqrt(a*(a-4))
lm, lp = (a+2-delta)/2, (a+2+delta)/2
wm = (1.5-lm)/delta
cov = wm*np.exp(-lm*t)+(1-wm)*np.exp(-lp*t)
peak = (np.sqrt(2*a+1)-3)/2
reset_cov = np.exp(-peak*t)*cov
axes[0].plot(t, 100*cov, color=COLORS[0], label="No resetting")
axes[0].plot(t, 100*reset_cov, color=COLORS[1], linestyle="--", label=r"Reset at $r_{\mathrm{loc}}$")
axes[0].axhline(0, color="0.5", linewidth=0.65)
axes[0].set(xlabel="Time t", ylabel=r"Scaled covariance $10^2 C_{e_1}(t)$", xlim=(0.15, 2.0))
axes[0].legend(frameon=False, loc="lower right")
theta = 2*np.pi/12
lam, omega = 1-np.cos(theta), np.sin(theta)
r = np.linspace(0, 2.5, 1201)
vr = 2*(lam+r)/((lam+r)**2+omega**2)
v0 = 2*lam/(lam**2+omega**2)
rp, rr = omega-lam, omega**2/lam-lam
axes[1].plot(r, vr/v0, color=COLORS[2])
axes[1].axhline(1, color="0.55", linewidth=0.65)
axes[1].axvline(rp, color=COLORS[0], linewidth=0.9, linestyle=":", label="Mode maximum")
axes[1].axvline(rr, color=COLORS[1], linewidth=0.9, linestyle="--", label="Baseline recovery")
axes[1].plot([rp, rr], [1/omega/v0, 1], "o", color=COLORS[2], ms=3.5)
axes[1].set(xlabel="Reset rate r", ylabel="Relative first-mode variance", xlim=(0, 2.5), ylim=(0.5, 2.15))
axes[1].legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1, loc="upper right")
for ax, letter in zip(axes, "ab"):
    panel(ax, letter)
save(fig, "Fig2")
table("Fig2a.csv", ["time", "covariance_no_reset", "covariance_reset_at_local_threshold"], [t, cov, reset_cov])
table("Fig2b.csv", ["reset_rate", "relative_first_mode_variance"], [r, vr/v0])
print("Wrote Fig1.pdf, Fig2.pdf, EPS equivalents and four analytical CSV tables.")
