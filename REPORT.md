# Modular grokking and flow reproduction report

## Status

Repository scaffolding and implementation are complete. A three-operation smoke
training run is the only authorized execution at this stage. Full-size training is
scale-gated, and no trained-checkpoint flow extraction or numbered-definition analysis
has been run.

## 1. What is being reproduced

The source repository is `safe-flow-mech-interp-lab` at commit `d599d2f` (20 August
2026). The implementation relevant to this project entered in commit `3cf3870`, which
added subtraction and multiplication and froze the universality/forecasting protocol.
The later commits primarily added analyses, controls, figures, and artifacts.

Each run learns one operation:

- addition: `(a + b) mod p`;
- subtraction: `(a - b) mod p`;
- multiplication: `(a * b) mod p`, restricted to nonzero residues so multiplication
  becomes addition after a discrete-log remapping.

“Multi-task” here means a controlled matrix of separate add/sub/mul runs. It does not
mean one model jointly trained on all three labels. Keeping tasks separate matches the
source experiment and makes cross-operation comparisons interpretable.

The canonical model has one causal attention block, four attention heads, residual
width 128, a 512-unit ReLU MLP, learned token/position embeddings, no layer norm, and
an untied output matrix. It sees `[a, b, =]` and predicts the answer at the last token.
Training uses a fixed random subset of the complete operation table, full-batch AdamW,
and enough weight decay and training time for memorization to precede generalization.

## 2. Grokking from first principles

Ordinary generalization improves gradually while training accuracy and test accuracy
rise together. In grokking, the model first memorizes its training examples: training
accuracy becomes nearly perfect while test accuracy stays low. Much later, test
accuracy rises sharply. The operational checkpoint used in the source study was the
first saved point with at least 99% training accuracy and 90% test accuracy.

This lab records loss, accuracy, and an operation-matched Fourier fraction at every
saved checkpoint. The Fourier quantity asks whether the learned logits concentrate on
frequency patterns expected for the correct modular operation. It is a behavioral/
representational baseline and does not use flow calculations.

## 3. Findings from the source repository

These are prior findings to reproduce, not findings newly established by this lab:

1. In the frozen 27-run study, 24 runs grokked within 30,000 epochs. Three subtraction
   runs did not.
2. A maximal-path contraction criterion passed 7/9 addition runs, 8/9 multiplication
   runs, and 0/6 evaluable subtraction runs. Its preregistered universality claim
   therefore failed.
3. The contraction was transition-specific under matched windows: 15/24 transitions,
   0/24 post-grok windows, and 2/17 memorization windows passed the full criterion.
4. Numbered flow features improved leave-one-run-out prediction of log grokking time
   from MAE 0.157 to 0.132 relative to the complete non-certificate baseline, but did
   not beat a trivial predictor when an entire hyperparameter cell was held out.
5. A later, explicitly post-hoc all-subpath reanalysis reversed the simple contraction
   story: its exclusive-packing quantity increased at the transition in every one of
   21 evaluable runs across all operations. The best current hypothesis is route
   consolidation, but it requires fresh seeds for confirmation.
6. A length-weighted integral was not a universal transition marker. Addition,
   subtraction, and multiplication followed different trajectories.
7. None of these observations demonstrates that a route is causally necessary in the
   Transformer. The flow graph is a positive attribution construction with attention
   probabilities held fixed; query/key paths are outside its scope.

## 4. Reproduction design

Training is isolated from interpretation:

1. `plan` expands the experiment matrix and estimates optimizer steps, checkpoint
   count, weight storage, and future flow-graph count.
2. `train` creates only datasets, model checkpoints, behavior metrics, manifests, and
   Markdown run reports. Large matrices require an explicit scale flag.
3. `extract-flows` is offline. For fixed model attention probabilities, it decomposes
   positive support/opposition for the target and strongest competitor. It stores raw
   edge marginals and canonical path weights before any candidate definition is used.
4. `summarize` reads only those raw records and computes `Definition-01` through
   `Definition-04`. This means later definitions can be added without retraining or
   rerunning model attribution.

Checkpoints contain weights and behavior metrics. A single `latest.pt` additionally
contains optimizer and scheduler state for resumption and is overwritten, preventing
optimizer-state multiplication across dense checkpoints. Every immutable checkpoint
gets a byte count and SHA-256 digest.

Raw data is one compressed JSONL file per checkpoint. Each successful graph record
contains run/checkpoint/example identifiers, operands, target, competitor, flow kind,
conservation error, every `[tail, head, value]` edge, and every canonical node path with
its weight. Node labels live once in the run manifest. Degenerate polarities are stored
as explicit status records rather than silently discarded.

## 5. Candidate formulas

Names are deliberately withheld until comparative evidence exists.

### Definition-01

At each ambiguous internal junction, add the positive lower bounds for adjacent edge
pairs, then divide by total throughput across those junctions.

### Definition-02

Add the lower-bound values of maximal eligible paths and divide by source flow. Since
paths can overlap, this is not asserted to be a union-mass lower bound.

### Definition-03

Enumerate all eligible paths and greedily retain paths that cannot occur on the same
complete source-to-sink route. Add their lower bounds and divide by source flow. This
is a conservative lower bound for the selected union.

### Definition-04

Start with source flow; for every eligible path, add its lower bound times edge length;
then divide by source flow. Nested and overlapping paths both contribute, so this is an
integral rather than a union-mass lower bound.

All result columns retain only the numbered identifiers.

## 6. Scale gate

The compact anchor matrix in `configs/anchor_template.toml` is:

- 3 operations x 5 seeds = 15 runs;
- modulus 113 and canonical model dimensions;
- 30,000 epochs per run, or 450,000 full-batch optimizer steps total;
- checkpoint every 100 epochs plus six early points, giving 307 checkpoints per run
  and 4,605 immutable model checkpoints total;
- eight fixed train and eight fixed test examples, four flow kinds, yielding 294,720
  raw graphs if every checkpoint is analyzed.

This increases seed count from three to five per selected cell and temporal resolution
from 250 to 100 epochs. The exact selected-cell reproduction in
`configs/scale_template.toml` uses the nine source-study cells and five seeds: 45 runs,
1,350,000 optimizer steps, 13,815 checkpoints (about 11.7 GiB of weights before
serialization overhead), and 884,160 possible raw graphs. Repeating the complete 3x3
hyperparameter grid would be larger still and should be budgeted separately.

Before full training, confirm one of these scopes:

- anchor: 15 runs above;
- expanded: add selected weight-decay/train-fraction cells after a behavioral pilot;
- custom: a different seed count, checkpoint interval, modulus, or epoch budget.

Before the flow phase, separately confirm checkpoint subset, examples per split, and
the Hugging Face dataset destination. `/workspace` is not volume-backed on this
instance, so approved valuable artifacts should be uploaded before instance recycle or
destruction.

## 7. Validation record

The intended validation sequence is formatting/lint, all synthetic unit tests, the
three-operation smoke training run, checkpoint reload/digest checks, and a final test
pass. Synthetic flow tests validate conservation and formula plumbing; they do not
analyze a trained checkpoint and do not constitute the paused flow experiment.

Completed on the RTX 5060 Ti instance:

- lint and formatting checks passed;
- 20 unit tests passed in 2.81 seconds;
- smoke training completed 20 epochs for add, sub, and mul;
- 21/21 immutable checkpoints matched their SHA-256 manifests and reloaded with
  `weights_only=True`;
- the complete smoke artifact directory is 1.4 MiB;
- final train/test accuracies were 0.140/0.025 for add, 0.200/0.050 for sub, and
  0.163/0.069 for mul;
- as expected, this 20-epoch plumbing run did not grok and supports no scientific
  conclusion;
- the unconfirmed large-run command was tested and refused execution;
- no `flow-artifacts` directory exists, and all run protocols retain flow status
  `not_run`.
