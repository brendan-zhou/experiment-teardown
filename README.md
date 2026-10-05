# Experiment Teardown: Smart Refill Nudges for EnlivenHealth

**What this is:** a full experiment design + readout for a (hypothetical) EnlivenHealth
feature — AI-timed prescription refill reminders. The trial data is **simulated**;
the value here is the design rigor and readout discipline, which is exactly what
senior data analyst interviews test.

**Why this product:** EnlivenHealth is Omnicell's patient engagement platform —
medication adherence outreach, refill reminders, patient messaging for health
systems and pharmacies. Refill behavior is both a patient-outcome metric and a
revenue metric, which makes the metric design interesting.

---

## 1. The metric tree

North star: **medication adherence** (proportion of days covered, PDC) — it's what
the product exists to improve, and it's what health-system buyers put in contracts.

But you can't run a 2-month experiment on PDC directly (it needs 12 months of
fill history to move). So the tree decomposes into leading indicators:

```
PDC (north star, 12-mo)
 └── Refill-within-7d rate  ← PRIMARY metric for this test
      ├── Nudge open / tap-through rate   (did the message land?)
      ├── Time-to-refill after due date   (how fast did it work?)
      └── Refill revenue per enrolled patient (the revenue lens)
Guardrails (must not regress):
 ├── Message opt-out rate      (trust is the inventory here)
 └── Support tickets per 1k patients (operational load)
```

**Why refill-within-7d and not raw refill count:** counts scale with enrollment,
rates don't. And the 7-day window matches the clinical logic — a refill 30 days
late is nearly as bad as a missed one for adherence.

## 2. The experiment design

**Hypothesis:** reminders timed to each patient's historically most-responsive
hour will lift refill-within-7d rate vs. the current fixed 9am batch send.

**Randomization unit: the pharmacy, not the patient.** Patients in one pharmacy
share staff, workflows, and word-of-mouth — randomizing patients would leak the
treatment into control through staff behavior change. Cluster randomization is
the right call, and it's expensive: with ICC=0.02 and ~1,600 patients per
pharmacy over 2 months, the design effect is **33x**. A naive power calc says
you need ~9k patients/arm; the honest one says ~300k/arm, i.e. **189 pharmacies
per arm**.

That number kills most pilots, so the design buys back power two ways
(quantified in `scripts/01_design.py`):
1. **Pair-match** pharmacies on pre-period refill rate and size, then adjust for
   the pre-period rate in analysis (ANCOVA). With pre-post correlation ρ=0.6,
   variance drops 36% → **~121 pharmacies/arm, 242 total**.
2. The power curve (`charts/power_curve.png`) shows what happens if the true
   effect undershoots the 2pp MDE — shared with stakeholders *before* launch so
   nobody is surprised by a null.

**MDE = +2pp** on a 62% base rate. Smaller than that isn't worth the operational
cost of retraining the timing model; larger is unrealistic for a messaging tweak.

## 3. The readout (simulated trial, n=242 pharmacies)

| Metric | Effect | 95% CI | p |
|---|---|---|---|
| **Refill-within-7d rate** | **+2.65pp** | +1.41 to +3.89 | <0.0001 |
| Opt-out rate | +0.35pp (1.50% → 1.85%) | +0.29 to +0.41 | <0.0001 |
| Support tickets / 1k | −0.39 | −0.82 to +0.04 | 0.078 |
| New patients (<6mo) | +4.91pp | +3.57 to +6.25 | <0.0001 |
| Existing patients | +1.90pp | +0.66 to +3.13 | 0.003 |

SRM check: PASS (302,400 vs 302,400 patients, χ² p=1.0).

Three things worth noting about the analysis itself:
- **Inference is at the cluster level** (t-test on pharmacy rates). The naive
  patient-level test gives p≈0 — standard errors ~5.7x too small. Getting this
  wrong is the most common way to ship a false positive from a cluster trial.
- **The guardrail fired.** Opt-outs rose a statistically significant +0.35pp.
  Significant ≠ important — but it needs a decision, not a shrug (see below).
- **Heterogeneity is the story:** new patients respond 2.5x more strongly. The
  average effect (+2.65pp) understates the value for the segment that matters
  most for growth.

**Revenue lens:** +26.5 incremental refills per 1,000 enrolled patients over
2 months. Revenue = incremental refills × gross profit per refill — the formula
is in the script; I don't assert a margin number that isn't mine to assert.

## 4. Decision: SHIP — with conditions

**Ship.** The primary metric cleared the pre-registered MDE with room to spare,
the SRM check passed, and support load didn't move. But the opt-out increase is
real, and opt-outs are *durable* — a patient who opts out is gone from every
future campaign, which compounds against the adherence north star.

Conditions:
1. **Frequency cap**: max 2 nudges per refill cycle. The treatment arm had no
   cap; capping should trim the opt-out lift while keeping most of the refill
   gain (the first nudge does the work — verify in a follow-up).
2. **Roll out to new patients first**, where the effect is 2.5x larger and the
   adherence stakes are highest (early therapy months predict long-term PDC).
3. **Monitor opt-outs as a rollout guardrail** for 90 days post-launch, with a
   pre-committed rollback threshold (e.g., opt-out rate >2.5% sustained).

What I'd run next: a 2×2 on send-time personalization × frequency cap to find
the frontier, and a 12-month holdout cohort to confirm the PDC north star
actually moves — because refill-within-7d is a proxy, and proxies need auditing.

## 5. Reproduce it

```bash
pip install -r requirements.txt
python scripts/01_design.py    # power analysis + design.json + power curve
python scripts/02_readout.py   # simulated trial + effects.csv + readout charts
```

All randomness is seeded (`rng = np.random.default_rng(42)`) — the readout
reproduces exactly. To be explicit once more: the *trial data* is simulated;
the *methods* (cluster power analysis, SRM check, cluster-level inference,
guardrail framework) are the real artifact.
