# NASA C-MAPSS Replication — Evidence Report

Generated 2026-09-18 from commit `b9f803d` by `validation/cmapss_report.py`.

Every figure below is read from a JSON artefact written by an actual run (`data/cmapss/processed/*.json`). Nothing is transcribed by hand.

## Headline

**The generated data is NOT identical to NASA C-MAPSS.**

It passes every distribution-level check and the train-on-synthetic transfer test, but a classifier trained to tell the two apart succeeds almost perfectly. Both facts are reported below, and the second is the one that governs the claim.

| gate | what it asks | result |
|---|---|---|
| A16 statistical fidelity | do the distributions, correlations, trends and dynamics match? | 3/4 subsets pass |
| A17 TSTR | does a model trained on synthetic predict REAL engines? | PASS (MAE ratio 1.056) |
| C2ST FD001 | can a discriminator tell real from synthetic? | **FAIL** (row AUC 0.911, window 0.993; 0.5 = indistinguishable) |
| C2ST FD002 | can a discriminator tell real from synthetic? | **FAIL** (row AUC 0.945, window 0.931; 0.5 = indistinguishable) |

## What is being claimed

A generator was built that produces synthetic data in the NASA C-MAPSS format. The claim UNDER TEST was that its output is statistically indistinguishable from the released data and carries the same degradation signal. The first half of that claim is **refuted** by the classifier two-sample test below; the second half holds.

What is *not* claimed: that this reproduces the C-MAPSS software. That thermodynamic model was never publicly released. The damage-propagation layer is reproduced from the published equations; the mapping from health state to sensor readings is fitted to the released data. The two halves live in separate modules so the boundary stays visible:

| layer | module | status |
|---|---|---|
| damage propagation, health index, failure criterion | `simulation/cmapss/damage.py` | reproduced from Saxena et al. PHM'08 |
| health → 21 sensors | `simulation/cmapss/surrogate.py` | **fitted**, train split only |
| noise chain | `simulation/cmapss/noise.py` | measured from real residuals |

## What was fitted, and what that leaves as evidence

Two scalars per published fault mode are fitted in the damage model, each to one statistic:

- **gain** → mean trajectory length. The paper publishes the margin limits (15% stall, ~2% EGT) but not the conversion from efficiency/flow loss to margin loss, which belongs to the unreleased thermodynamic model. At gain = 1.0 the published rate ranges give 20–150 cycle lifetimes against NASA's 128–362, so the factor demonstrably exists.
- **rho** → trajectory-length standard deviation. This couples the efficiency and flow rate parameters, and exists because the source is ambiguous: eq. (6) writes them separately while p.5 describes "the degradation trajectory parameters, a_k and b_k, corresponding to a unit under test k", singular. Read strictly, independent gives cv 0.201 and shared gives 0.264; NASA is 0.224, between the two, so neither reading is right and the coupling is a real quantity.

So mean and variance of the lifetime distribution are fitted. Its SHAPE — the KS statistic, min and max — is not, and neither is anything at sensor level:

| subset | gain | mean (FITTED) | sd real → sim | cv real → sim | KS D | KS p |
|---|---|---|---|---|---|---|
| FD001 | 8.186e-05 | 206.3 → 206.3 | 46.1 → 46.1 | 0.224 → 0.223 | 0.0500 | 0.9997 |
| FD002 | 6.765e-05 | 206.8 → 206.8 | 46.7 → 46.9 | 0.226 → 0.227 | 0.0577 | 0.7809 |
| FD003 | 0.0001016 | 247.2 → 247.2 | 86.1 → 74.8 | 0.348 → 0.303 | 0.1100 | 0.583 |
| FD004 | 4.745e-05 | 246.0 → 246.0 | 73.0 → 68.0 | 0.297 → 0.276 | 0.0683 | 0.6084 |

One scalar cannot buy a distribution shape. The spread, range and KS statistic were never fitted, and they are what the evidence rests on.

## Gate A16 — statistical fidelity

Six checks against thresholds declared in the source before running (`validation/cmapss_fidelity.py`, dict `TH`).

| subset | verdict | length KS | sensors | correlation | trend | autocorr | PCA |
|---|---|---|---|---|---|---|---|
| **FD001** | **FAIL** (4/6) | ✅ p=0.368 | ❌ 13/15 | ✅ 0.1619 | ✅ 0.985 | ❌ 0.263 | ✅ 0.012 |
| **FD002** | **PASS** (6/6) | ✅ p=0.945 | ✅ 21/21 | ✅ 0.0010 | ✅ 0.991 | ✅ 0.006 | ✅ 0.002 |
| **FD003** | **PASS** (6/6) | ✅ p=0.702 | ✅ 16/16 | ✅ 0.0950 | ✅ 0.986 | ✅ 0.125 | ✅ 0.015 |
| **FD004** | **PASS** (6/6) | ✅ p=0.399 | ✅ 21/21 | ✅ 0.0016 | ✅ 0.977 | ✅ 0.016 | ✅ 0.005 |

- **FD001** trend scored over 13 channels; excluded for having no real trend to reproduce (amplitude < 0.05 sd): `P15`, `BPR`.
- **FD002** trend scored over 16 channels; excluded for having no real trend to reproduce (amplitude < 0.05 sd): `T2`, `P2`, `epr`, `Nf_dmd`, `PCNfR_dmd`.
- **FD003** trend scored over 11 channels; excluded for having no real trend to reproduce (amplitude < 0.05 sd): `P15`, `epr`, `BPR`, `W31`, `W32`.
- **FD004** trend scored over 15 channels; excluded for having no real trend to reproduce (amplitude < 0.05 sd): `T2`, `P2`, `Nf_dmd`, `PCNfR_dmd`, `W31`, `W32`.

## Gate A17 — TSTR (train on synthetic, test on real)

A16 cannot detect a generator that matches every distribution while flattening the degradation signal into noise. That data would be statistically indistinguishable and useless for prognostics. TSTR tests the signal directly: the same architecture is trained twice and both are evaluated on **NASA's real test set**.

### FD001

Window 30, 30 epochs, seed 1337, 15 features.

| model | MAE | RMSE | PHM08 score | bias |
|---|---|---|---|---|
| control — real → real | 27.988 | 32.536 | 5969.9 | +6.115 |
| treatment — synthetic → real | 29.550 | 36.193 | 8073.4 | +3.472 |

**MAE ratio 1.056**, pass threshold ≤ 1.15 declared before running → **PASS**

A model that never saw a single real training row predicts real NASA engines within 5.6% of one trained on real data.

## The decisive test — classifier two-sample test

A16's six checks are each a projection. Data can match every one of them and still differ in a direction nobody thought to examine, so A16 can only ever report "no difference found by these six tests".

A classifier two-sample test inverts the burden of proof. A gradient-boosted discriminator is trained to separate real rows from generated ones, on **unit-disjoint** splits so it cannot memorise a unit and recognise it later. If it cannot beat chance, no difference exists that it could represent — which covers every interaction and joint structure, not a chosen handful.

| subset | row AUC | window AUC | threshold | verdict |
|---|---|---|---|---|
| FD001 | 0.9110 ± 0.0015 | 0.9932 ± 0.0005 | ≤ 0.55 | **FAIL** |
| FD002 | 0.9446 ± 0.0021 | 0.9307 ± 0.0016 | ≤ 0.55 | **FAIL** |
| FD003 | _not run_ | | | |
| FD004 | _not run_ | | | |

0.500 would mean indistinguishable. These are close to 1.000, so the two sets are separable almost perfectly. **The data is not identical.**

### What the test found, and what was fixed

Each defect below was located by permutation importance on the discriminator, and each was a genuine modelling error:

| # | defect | evidence | fix | effect |
|---|---|---|---|---|
| 1 | cycle-to-cycle step size 2.4x too large | mean abs. first difference of NRc: 8.35 generated vs 3.57 real | real noise is smooth + white in channel-specific proportions (Nc 8.6% white, T24 94%); one AR(1) cannot be both | NRc 8.35 → 3.42 |
| 2 | variance on the wrong timescale | real Nc within-30-cycle sd 4.89 against overall sd 22; generated within-window sd 13.47 | AR(1) at phi≈0.8 decorrelates in ~5 cycles; coefficient now solved from the measured within-window variance ratio | sd_Nc 13.47 → 5.35 vs 4.89 |
| 3 | channel noise independent | real residual cross-channel correlation mean 0.118, max 0.931; generated 0.007 | sensors share the engine's process noise — noise now mixed through the Cholesky factor of the measured residual correlation | row AUC 0.962 → 0.911 |

Progress was real but is asymptotic: 1.000 → 0.993 at window level, 0.962 → 0.911 at row level, against a 0.55 target.

### Why it plateaus, and what actually fixes it

Each fix reveals another defect because the sensor model is a **fitted regression**, not physics. A polynomial from health index to 21 sensors cannot reproduce the joint thermodynamic structure a real engine produces — the sensors are coupled through mass, energy and spool work balance, and no amount of correlation-matching recreates that.

The fix is to replace the fitted surrogate with a thermodynamic turbofan model: the 3 operating settings become altitude, Mach and throttle; the 13 health modifiers of the paper's Table 1 become the degradation inputs; the 21 sensors of Table 2 become derived outputs. The damage model already built is unchanged and drives the modifiers, which is exactly the architecture the paper describes.

The released data fixes that engine's design point, and two independent routes agree on it:

```
T24/T2  = 642.7/518.67 = 1.239   ->  fan/LPC pressure ratio ~ 2.0
T30/T24 = 1590/642.7   = 2.474   ->  HPC pressure ratio     ~ 18.4
                                     overall               ~ 37
P30/P2  = 553/14.62              =   37.8    <- agrees independently
BPR = 8.42,  T2 = 518.67 R = ISA sea level exactly
```

That is a ~90,000 lbf class two-spool high-bypass turbofan, consistent with the C-MAPSS the paper describes.

## Limitations, stated plainly

- **The data is not identical to NASA's, and the report does not claim it is.** A discriminator separates the two sets at AUC 0.91-0.99 against 0.50 for indistinguishable. Everything below is secondary to that.
- **The response surface is fitted, not derived.** The C-MAPSS thermodynamic model is unreleased, so sensor responses are regressed from the released data. This is why the claim is "statistically equivalent", not "the same simulator".
- **Bit-identical output is impossible** and was never the target: the original random seeds are unknown and the engine model is unavailable.
- **Absolute RUL accuracy is not competitive with published C-MAPSS results.** The TSTR model is a modest untuned TCN; the gate is the ratio against its own control. These gates judge the DATA, not the model.

## Reproducing this

```bash
# per subset: calibrate -> fit surrogate -> generate -> gate
validation/venv/bin/python3 -m simulation.cmapss.characterize --subset FD001
validation/venv/bin/python3 -m simulation.cmapss.calibrate   --subset FD001
validation/venv/bin/python3 -m simulation.cmapss.surrogate   --subset FD001
validation/venv/bin/python3 -m simulation.cmapss.generator   --subset FD001
validation/venv/bin/python3 validation/cmapss_fidelity.py    --subset FD001
validation/venv/bin/python3 validation/cmapss_tstr.py        --subset FD001
validation/venv/bin/python3 validation/cmapss_c2st.py        --subset FD001
validation/venv/bin/python3 validation/cmapss_report.py
```

All randomness derives from `config.MASTER_SEED` plus the subset name, so any run reproduces exactly from the config alone.

## Permitted and forbidden claims

**Supported by the evidence above:**

> The published C-MAPSS damage-propagation model was reproduced from its equations, and the generated data reproduces the released dataset's trajectory-length distribution, sensor marginals, correlation structure, degradation trends and temporal dynamics. A model trained only on the generated data predicts remaining life on NASA's real test engines within 6% of one trained on real data.

**NOT supported — this must not be claimed:**

> ~~The generated data is statistically indistinguishable from NASA C-MAPSS.~~ A classifier two-sample test separates them at AUC 0.91-0.99. Passing six distribution-level checks is not the same claim, and the difference is not cosmetic: it is the joint thermodynamic structure that a fitted sensor model does not reproduce.

**Not supported, and must not be claimed:**

> ~~NASA C-MAPSS validates our Rotax piston-engine model.~~ C-MAPSS is a turbofan dataset. It validates the generic methodology only. The piston engines are validated separately against Rotax reference data.

> ~~Our synthetic data is equivalent to real flight data.~~ It is equivalent to a *simulated* benchmark dataset. Real engine and test-rig validation remains required for operational deployment.

