"""02_readout.py — simulated readout of the refill-nudge cluster trial.

IMPORTANT: the trial data here is SIMULATED (seeded). The portfolio value is
in the design rigor and the readout discipline, not in real patient data.
The simulation is built to contain the situations that make readouts hard:
  * a significant primary-metric win (+2.5pp refill rate),
  * a significant guardrail movement (opt-outs +0.4pp),
  * heterogeneous effects (new patients respond much more),
so the ship/no-ship reasoning in the README is about a genuine tradeoff.

Method notes:
  * Pharmacy-level true rates drawn with ICC=0.02 between-pharmacy variance.
  * Patients aggregated per pharmacy via binomial draws (vectorized).
  * Inference at the CLUSTER level (t-test on pharmacy rates) — the naive
    patient-level test would understate SEs by ~sqrt(design effect) and is
    shown only as a cautionary contrast.
  * SRM check on randomized patient counts.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(42)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
CHARTS = os.path.join(HERE, "charts")

with open(os.path.join(RESULTS, "design.json")) as f:
    DESIGN = json.load(f)

N_ARM = DESIGN["pharmacies_per_arm"]          # 121 / arm (matched design)
M = DESIGN["patients_per_pharmacy_per_month"] * DESIGN["months"]  # 1600
BASE = DESIGN["base_rate"]                     # 0.62
ICC = DESIGN["icc"]

# True effects baked into the simulation
EFF_OVERALL = 0.025     # +2.5pp refill rate (above the 2pp MDE: detectable)
EFF_NEW = 0.045         # new patients (<6mo enrolled) respond more
EFF_EXISTING = 0.018
OPTOUT_BASE, OPTOUT_EFF = 0.015, 0.004   # guardrail: opt-outs rise 1.5% -> 1.9%
TICKET_BASE = 8 / 1000                   # support tickets per patient; no effect

# ---------- simulate ----------
# pair-matched: pharmacies come in pairs with similar base rates
sd_between = np.sqrt(ICC * BASE * (1 - BASE))
pair_base = rng.normal(BASE, sd_between, N_ARM).clip(0.3, 0.9)

rows = []
for i, pb in enumerate(pair_base):
    for arm, eff in (("control", 0.0), ("treatment", None)):
        n_new = rng.binomial(M, 0.25)
        n_exist = M - n_new
        if arm == "treatment":
            r_new = np.clip(pb + EFF_NEW, 0, 1)
            r_exist = np.clip(pb + EFF_EXISTING, 0, 1)
            optout = OPTOUT_BASE + OPTOUT_EFF
        else:
            r_new = r_exist = pb
            optout = OPTOUT_BASE
        ref_new = rng.binomial(n_new, r_new)
        ref_exist = rng.binomial(n_exist, r_exist)
        rows.append({
            "pharmacy_id": f"P{i:03d}_{arm[0]}", "arm": arm, "pair": i,
            "n_patients": M, "n_new": n_new,
            "refills": ref_new + ref_exist,
            "refill_rate": (ref_new + ref_exist) / M,
            "refills_new": ref_new,
            "optouts": rng.binomial(M, optout),
            "tickets": rng.poisson(M * TICKET_BASE),
        })
pharm = pd.DataFrame(rows)
pharm["optout_rate"] = pharm["optouts"] / pharm["n_patients"]
pharm["tickets_per_1k"] = pharm["tickets"] / pharm["n_patients"] * 1000
pharm["refill_rate_new"] = pharm["refills_new"] / pharm["n_new"]
pharm["refills_exist"] = pharm["refills"] - pharm["refills_new"]
pharm["refill_rate_exist"] = pharm["refills_exist"] / (pharm["n_patients"] - pharm["n_new"])
pharm.to_csv(os.path.join(RESULTS, "readout_pharmacy_level.csv"), index=False)

ctrl = pharm[pharm.arm == "control"]
trt = pharm[pharm.arm == "treatment"]

# ---------- 1. SRM check ----------
total = pharm["n_patients"].sum()
exp_half = total / 2
chi2 = ((pharm.groupby("arm")["n_patients"].sum() - exp_half) ** 2 / exp_half).sum()
srm_p = 1 - stats.chi2.cdf(chi2, df=1)
print(f"[SRM] patients control={ctrl['n_patients'].sum():,} "
      f"treatment={trt['n_patients'].sum():,}  chi2 p={srm_p:.3f} "
      f"-> {'PASS' if srm_p > 0.01 else 'FAIL: investigate randomization'}")

# ---------- 2. primary metric (cluster-level) ----------
def cluster_test(metric):
    c, t = ctrl[metric], trt[metric]
    est = t.mean() - c.mean()
    t_stat, p = stats.ttest_ind(t, c, equal_var=False)
    se = est / t_stat if t_stat != 0 else np.nan
    ci = (est - 1.96 * se, est + 1.96 * se)
    return est, se, ci, p

results = []
est, se, ci, p = cluster_test("refill_rate")
results.append(("Primary: refill-within-7d rate", est, se, ci, p))
print(f"\n[PRIMARY] refill rate lift: {est:+.2%} "
      f"(95% CI {ci[0]:+.2%} to {ci[1]:+.2%}), p={p:.4f}")

# cautionary contrast: naive patient-level test (WRONG for cluster trials)
x_c, n_c = ctrl["refills"].sum(), ctrl["n_patients"].sum()
x_t, n_t = trt["refills"].sum(), trt["n_patients"].sum()
z = ((x_t / n_t - x_c / n_c)
     / np.sqrt((x_t + x_c) / (n_t + n_c) * (1 - (x_t + x_c) / (n_t + n_c)) * (1 / n_t + 1 / n_c)))
p_naive = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"  (naive patient-level p would be {p_naive:.2e} — "
      f"~{1/np.sqrt(33):.2f}x the true SE. Never do this.)")

# ---------- 3. guardrails ----------
for name, col, fmt in [("Guardrail: opt-out rate", "optout_rate", "+.2%"),
                       ("Guardrail: tickets / 1k pts", "tickets_per_1k", "+.2f")]:
    est, se, ci, p = cluster_test(col)
    results.append((name, est, se, ci, p))
    print(f"[{name.split(':')[0].upper()}] {name.split(': ')[1]} change: "
          f"{est:{fmt}} (95% CI {ci[0]:{fmt}} to {ci[1]:{fmt}}), p={p:.4f}")

# ---------- 4. segments ----------
for name, col in [("Segment: new patients (<6mo)", "refill_rate_new"),
                  ("Segment: existing patients", "refill_rate_exist")]:
    est, se, ci, p = cluster_test(col)
    results.append((name, est, se, ci, p))
    print(f"[SEGMENT] {name.split(': ')[1]} lift: {est:+.2%} "
          f"(95% CI {ci[0]:+.2%} to {ci[1]:+.2%}), p={p:.4f}")

eff_df = pd.DataFrame(results, columns=["metric", "estimate", "se", "ci95", "p_value"])
eff_df.to_csv(os.path.join(RESULTS, "effects.csv"), index=False)

# ---------- 5. revenue lens ----------
inc_refills_per_1k = eff_df.loc[0, "estimate"] * 1000
print(f"\n[REVENUE LENS] incremental refills: {inc_refills_per_1k:.1f} per 1,000 "
      f"enrolled patients over 2 months")
print("  revenue = incremental refills x gross profit per refill "
      "(plug in your formulary margin; not asserted here)")

# ---------- charts ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
labels = [r[0].split(": ")[1] for r in results]
ests = [r[1] * (100 if "tickets" not in r[0] else 1) for r in results]
los = [(r[1] - 1.96 * r[2]) * (100 if "tickets" not in r[0] else 1) for r in results]
his = [(r[1] + 1.96 * r[2]) * (100 if "tickets" not in r[0] else 1) for r in results]
colors = ["#4C78A8" if r[4] < 0.05 else "#9E9E9E" for r in results]
y = np.arange(len(labels))
ax.errorbar(ests, y, xerr=[np.array(ests) - np.array(los), np.array(his) - np.array(ests)],
            fmt="o", color="#333333", ecolor="#333333", capsize=4)
for i, c in enumerate(colors):
    ax.plot(ests[i], y[i], "o", color=c, markersize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("Estimated effect (pp; tickets per 1k for support)")
ax.set_title("Trial readout: primary win, guardrail cost, heterogeneous response")
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "readout_forest.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(ctrl["refill_rate"], bins=25, alpha=0.6, label="Control pharmacies", color="#9E9E9E")
ax.hist(trt["refill_rate"], bins=25, alpha=0.6, label="Treatment pharmacies", color="#4C78A8")
ax.axvline(ctrl["refill_rate"].mean(), color="#616161", linestyle="--")
ax.axvline(trt["refill_rate"].mean(), color="#1A3C5E", linestyle="--")
ax.set_xlabel("Pharmacy refill-within-7d rate")
ax.set_ylabel("Pharmacies")
ax.set_title("Cluster-level distribution: the unit of analysis is the pharmacy")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "pharmacy_distribution.png"), dpi=150)
plt.close(fig)
print("\nreadout saved -> results/effects.csv, results/readout_pharmacy_level.csv, charts/")
