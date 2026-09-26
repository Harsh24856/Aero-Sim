# validation_v5 - training and scoring the v5 models

Plan: `docs/v5_model_improvement_plan.md`. Physics: `backend/physics_v5.py`.
Data: `data/rotax_v5/<912|914|915|916>` (from `backend/generate_dataset_v5.py`).

## Run it

```
validation/venv/bin/python validation_v5/run_v5.py 914          # one engine, every step still to do
validation/venv/bin/python validation_v5/run_v5.py all          # 914, 912, 915, 916
validation/venv/bin/python validation_v5/run_v5.py 914 --force train   # retrain (and redo everything after)
```

Steps, each skipped if its output exists: `checks -> cache -> train -> calibrate -> rul -> test -> card`.
Outputs go to `artifacts/<engine>/`; the scorecard is `artifacts/<engine>/model_card.md`.
Training runs on the Metal GPU only (the device the models are served on); it stops if no GPU is visible.

Before the first engine, once: the hyper-parameter search on the 914 pilot
(its winner becomes the training config for every engine):

```
validation/venv/bin/python validation_v5/hp_search_v5.py --cache validation_v5/cache/pilot_914 --trials 24 --rungs 3,8,30
```

Notebooks: `validation/venv/bin/python validation_v5/make_notebooks_v5.py`, then open
`notebooks/` (index in `notebooks/README.md`). They call the same steps, so anything
`run_v5.py` already did loads instantly.

## Files

| File | Role |
|---|---|
| `run_v5.py` | The pipeline, step by step, resumable |
| `data_checks_v5.py` | 14 checks a dataset must pass before training |
| `pipeline_v5.py` | Window cache: scaled features, context, labels; per-epoch reshuffled windows |
| `model_architectures_v5.py` | The joint network (TCN + context; 6 heads; shared per-channel sensor head) |
| `train_v5.py` | Losses, training loop, selection by validation composite, calibration, save/load |
| `rul_v5.py` | Remaining life as a separate model on the network's own outputs |
| `score_v5.py` | Test scorecard: every head vs a do-nothing baseline, flight-bootstrap CIs, model card |
| `evaluate.py` | Metrics and the flight bootstrap (framework-free) |
| `gates_v5.py` | The quality targets and their check |
| `hp_search_v5.py` | Random search + successive halving |
| `observability_v5.{json,md}` | Which faults each engine's instruments can see (Phase 1) |
| `pilot_learnability.py` | Phase 2 pilot: are the labels learnable at all |
| `nb_helpers_v5.py`, `make_notebooks_v5.py` | The notebooks |
| `tests/test_pipeline_v5.py` | Cache matches raw data; windows deterministic; splits disjoint |

## Splits

Flights are split 60 / 15 / 10 / 15 into train / cal / val / test, stratified by
fault family. Train learns; cal fits the temperatures and the RUL model (flights the
network never trained on); val picks the epoch and the cut-offs; test is scored once.
