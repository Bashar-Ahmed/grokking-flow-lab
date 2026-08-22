# Grokking Flow Lab

A compact, standalone reproduction of modular add/subtract/multiply grokking,
with behavior-only training separated from raw flow extraction and numbered
candidate definitions.

## Quick start

```bash
./scripts/bootstrap.sh
.venv/bin/grokking-lab plan --config configs/smoke.toml
.venv/bin/grokking-lab train --config configs/smoke.toml --output runs/smoke
.venv/bin/grokking-lab report-behavior --runs runs/smoke
.venv/bin/pytest
```

Large runs are refused unless `--confirm-scale` is supplied. Flow extraction and
summarization independently require `--acknowledge-phase-gate`; do not cross that
gate until the training scale and flow protocol have been approved.

After approval, the two offline phases are:

```bash
.venv/bin/grokking-lab extract-flows \
  --run runs/<sweep>/<run-id> --output flow-artifacts/raw \
  --device cpu --workers 12 --resume --acknowledge-phase-gate

.venv/bin/grokking-lab summarize \
  --raw-run flow-artifacts/raw/<run-id> \
  --output flow-artifacts/definitions/<run-id> \
  --acknowledge-phase-gate
```

The raw format stores every edge marginal and every canonical path weight in one
gzip JSONL record per example/checkpoint/flow kind. It is intentionally retained
independently of derived definitions. Schema version 2 normalizes stored flow to
10,000,000 scaled units with four decimal subunits; divide by each record's
`flow_scale` to recover conventional unit flow. Every record includes `split` as
`train` or `test`.

To upload artifacts after credentials and destination are confirmed:

```bash
export HF_TOKEN=...
.venv/bin/grokking-lab upload --source flow-artifacts/raw \
  --repo-id <owner/dataset> --repo-type dataset --confirm-upload
```

The uploader creates private repositories by default. See [REPORT.md](REPORT.md)
for the experiment rationale, findings being reproduced, limitations, and scale gate.

Aligned train/test plots for every numbered definition can be generated after raw
extraction with:

```bash
uv pip install --python .venv/bin/python -e '.[plot]'
.venv/bin/python scripts/plot_aligned_definitions.py \
  --raw-root flow-artifacts/raw-main-50k \
  --behavior-summary runs/selected_cells_seed0_2_epoch50000/behavior_summary.csv \
  --output flow-artifacts/definition-aligned-plots --workers 12 \
  --min-source-epoch-exclusive 100 --resume
```

The strict source-epoch filter removes initialization checkpoints at epochs 0--100
from plots and endpoint summaries while retaining them in the raw derived data.

## Repository map

```text
configs/                 smoke config and gated scale proposal
scripts/bootstrap.sh     reuse the instance's Blackwell-compatible PyTorch
src/grokking_lab/        model, training, raw flow, numbered definitions, CLI
tests/                   dataset/model, checkpoint, flow, and formula tests
REPORT.md                plain-language research and execution report
```
