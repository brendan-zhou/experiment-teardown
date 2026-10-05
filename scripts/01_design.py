"""01_design.py — experiment design for the EnlivenHealth smart refill nudge test.

Context (fictionalized design exercise): EnlivenHealth, Omnicell's patient
engagement platform, wants to test AI-timed refill reminders (SMS/app push
sent at the patient-specific time they're most likely to act) against the
current fixed-schedule reminders.

Design decisions:
  * Randomization unit = pharmacy location (cluster randomization). Patients
    within a pharmacy share staff workflows and word-of-mouth; randomizing
    patients would contaminate the control group.
  * Primary metric: % of due refills completed within 7 days ("refill rate").
  * Guardrail metrics: message opt-out rate, support tickets per 1k patients.
    (A nudge that lifts refills but burns trust is a bad trade.)
  * Success metric for the revenue lens: refill revenue per enrolled patient.

This script:
  1. Computes the required number of pharmacy clusters via a power analysis
     for a cluster-randomized trial (accounts for intra-cluster correlation).
  2. Saves the design parameters that 02_readout.py will use to simulate
     the trial.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
CHARTS = os.path.join(HERE, "charts")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CHARTS, exist_ok=True)

# ---------- design parameters ----------
BASE_RATE = 0.62        # current refill-within-7d rate (control assumption)
MDE = 0.02              # minimum detectable effect: +2pp absolute
ALPHA = 0.05
POWER = 0.80
ICC = 0.02              # intra-cluster correlation across pharmacies
PATIENTS_PER_PHARMACY = 800   # eligible patients per location per month
MONTHS = 2              # trial duration

# ---------- power analysis for cluster-randomized trial ----------
# For a two-sample proportion test, per-group *patient* sample size first:
p1, p2 = BASE_RATE, BASE_RATE + MDE
p_bar = (p1 + p2) / 2
z_alpha = stats.norm.ppf(1 - ALPHA / 2)
z_beta = stats.norm.ppf(POWER)
n_patients_per_arm = (
    (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
     + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / MDE ** 2
)
# Inflate by the design effect for clustering:
m = PATIENTS_PER_PHARMACY * MONTHS
design_effect = 1 + (m - 1) * ICC
n_patients_per_arm *= design_effect
n_pharmacies_per_arm = int(np.ceil(n_patients_per_arm / m))

print(f"patients/arm needed (cluster-adjusted): {n_patients_per_arm:,.0f}")
print(f"design effect: {design_effect:.1f}x  (ICC={ICC}, {m} patients/location)")
print(f"-> pharmacies per arm: {n_pharmacies_per_arm}")
print(f"-> total pharmacies: {2 * n_pharmacies_per_arm}, run for {MONTHS} months")

# Reality check: 378 sites is a big ask for a pilot. Two standard ways to
# buy back power, quantified here so the tradeoff is explicit:
#   1. Pair-match pharmacies on pre-period refill rate + adjust for the
#      pre-period rate in analysis (ANCOVA). With pre-post correlation rho,
#      variance shrinks by (1 - rho^2).
#   2. lengthen the trial (more patients per cluster also raises the design
#      effect, so this helps less than intuition suggests).
RHO = 0.6  # assumed pre-post correlation of pharmacy refill rates
adj_factor = 1 - RHO ** 2
n_pharm_adj = int(np.ceil(n_pharmacies_per_arm * adj_factor))
print(f"\nwith pair-matching + pre-period adjustment (rho={RHO}): "
      f"~{n_pharm_adj} pharmacies/arm "
      f"({2 * n_pharm_adj} total) — variance x{adj_factor:.2f}")
print("design decision: pair-match on pre-period refill rate and pharmacy "
      "size; analyze with cluster-level regression on the adjusted rates.")

# ---------- power curve: what if the true effect is smaller/larger? ----------
def patients_per_arm_for_effect(eff):
    p2_ = BASE_RATE + eff
    p_bar_ = (BASE_RATE + p2_) / 2
    n = (
        (z_alpha * np.sqrt(2 * p_bar_ * (1 - p_bar_))
         + z_beta * np.sqrt(BASE_RATE * (1 - BASE_RATE) + p2_ * (1 - p2_))) ** 2
        / eff ** 2
    )
    return n * design_effect

effects = np.linspace(0.005, 0.04, 15)
powers = []
n_have = n_pharmacies_per_arm * m
for eff in effects:
    p2_ = BASE_RATE + eff
    # power with our fixed sample: invert the sample-size formula
    se = np.sqrt(design_effect * (BASE_RATE * (1 - BASE_RATE) + p2_ * (1 - p2_)) / n_have)
    powers.append(stats.norm.cdf(eff / se - z_alpha))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(effects * 100, powers, color="#4C78A8", linewidth=2.5)
ax.axhline(0.8, color="#E45756", linestyle="--", label="80% power target")
ax.axvline(MDE * 100, color="gray", linestyle=":", label=f"MDE = {MDE:.0%}")
ax.set_xlabel("True absolute lift in refill rate (pp)")
ax.set_ylabel("Statistical power")
ax.set_title("Power curve: cluster-randomized refill nudge test")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "power_curve.png"), dpi=150)
plt.close(fig)

design = {
    "base_rate": BASE_RATE, "mde": MDE, "alpha": ALPHA, "power": POWER,
    "icc": ICC, "patients_per_pharmacy_per_month": PATIENTS_PER_PHARMACY,
    "months": MONTHS, "pharmacies_per_arm": n_pharmacies_per_arm,
    "randomization_unit": "pharmacy_location",
    "primary_metric": "refill_within_7d_rate",
    "guardrails": ["opt_out_rate", "support_tickets_per_1k"],
}
with open(os.path.join(RESULTS, "design.json"), "w") as f:
    json.dump(design, f, indent=2)
print("design saved -> results/design.json, charts/power_curve.png")
