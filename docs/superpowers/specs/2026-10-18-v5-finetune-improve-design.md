# v5 Fine-tune Improvement: Design

Agreed in brainstorming, 2026-10-18 (approach A).

## Goal and success

- **Goal:** better live behaviour. An engine's new export ships only if it wins or ties `scripts/ab_v5.py` against today's live export, with its targeted fault improved, and no test-card gate flips from PASS to FAIL.
- **Budget:** one night (about 12 h) of machine time. The Mac stays plugged in with the lid open.
- **Method:** fine-tune the existing specialists. Do not retrain from scratch.

## What hurts live today (A/B 2026-10-16/17)

1. Air-filter fouling is never called on the 915 or 916 flights. On test, the 916's recall is 0.33.
2. Oil pump and oil degradation are swapped on the 912, 914 and 915.
3. RUL is dragged up by the W2 logbook trend on faults the logbook cannot see grow. The 915's turbo_high base RUL is 150 h (correct), but with the trend it hits the 222-235 h calendar cap.

## Components

1. **Gap measurement:**
   - A new `relabel_v5b.py --out KEY` writes a per-instrument relabel to a separate cache, so the live labels are never touched.
   - `gaps_v5.py <e>b --fault NAME --vs <KEY>` reports, for one fault, the share of test misses that the per-instrument rule calls invisible.
2. **Fine-tune mode** (`specialists_v5.py <e>b --finetune HEADS [--fault-weight name=w,...]`):
   - It loads the trained head (`Trainer.load(..., overrides)`).
   - It runs one low-learning-rate stage: all layers, `lr = min(1e-4, cfg lr / 5)` with a warm-up, 12 epochs, patience 3.
   - It uses `fit(keep_current=True)`, so the loaded weights stay unless validation improves.
   - Per-fault loss weights go through a new `fault_weights()` helper, in a weight separate from `fault_mask` (which severity also uses).
3. **Per-family RUL trend:**
   - `TrendGatedRUL(..., families, fam_cols)` uses the trend only where the window's argmax family is allowed and its probability is at least 0.5. `families=None` gives exactly today's output.
   - `fit_rul` adds a `+fam` candidate whose allowed families are those where the trend lowers validation wear-limited error for that family.
   - The existing validation rule picks among all candidates.

## Phases

| Phase | Work | Time |
|---|---|---|
| P0 | Build components 1-3 with TDD | about 1.5 h |
| P1 | Measure air filter (916, 915) and oil (912, 914, 915, 916); pick the paths | about 45 min |
| P2 | RUL refit, all 4 engines | about 1 h |
| P3 | Diagnosis fine-tunes (916, 915, 912, 914); the 916 is relabelled and gets all 5 heads if its air-filter misses are invisible | 3-5.5 h |
| P4 | Export, parity, e2e and A/B per engine | about 1.5 h |
| P5 | PR, merge, main checkout, restart, fly each changed engine, update `engine_info` scores | about 30 min |

## Out of scope

- Data regeneration and new architectures.
- Targeting severity bias: it is measured, not targeted.
