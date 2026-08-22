# Modular grokking and flow reproduction report

## Status

Repository scaffolding and implementation are complete. The nine-cell, three-seed,
50,000-epoch behavior-only training matrix is complete. No trained-checkpoint flow
extraction or numbered-definition analysis has been run.

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

The approved selected-cell reproduction in `configs/scale_template.toml` uses the nine
source-study cells, seeds 0--2, and 50,000 epochs: 27 runs, 1,350,000 optimizer steps,
13,689 checkpoints (about 11.6 GiB of weights before serialization overhead), and
876,096 possible raw graphs. This keeps the finer 100-epoch temporal resolution.

The 27-run training scope was approved on 22 August 2026. Flow extraction remains a
separate unapproved phase.

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
- 23 unit tests passed, including post-hoc/source isolation and exact serial/parallel
  flow-extraction equivalence coverage;
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

## 8. Completed 50,000-epoch behavior matrix

All 27 approved runs completed on 22 August 2026 in approximately two hours. Each run
has 507 saved checkpoints, for 13,689 total. A full integrity pass recomputed SHA-256
for every checkpoint and verified all 12,485,449,431 bytes against the manifests.

Using the frozen behavioral threshold—first saved checkpoint with training accuracy
at least 0.99 and test accuracy at least 0.90—26/27 runs grokked:

- addition: 9/9;
- subtraction: 8/9;
- multiplication: 9/9.

Two subtraction runs grokked only after the source study's 30,000-epoch budget:
`sub_frac0p25_wd1_seed2` at epoch 43,700 and `sub_frac0p3_wd1_seed1` at epoch
39,100. `sub_frac0p25_wd1_seed1` did not grok by epoch 50,000 and ended at 2.9%
test accuracy despite 100% training accuracy. The extended horizon therefore separates
late transitions from one still-censored run.

The complete plain-language behavior report and machine-readable table are stored in
`runs/selected_cells_seed0_2_epoch50000/REPORT.md` and `behavior_summary.csv`. These
results are behavioral only. Flow status remains `not_run` for every protocol.

## 9. Purely post-hoc 50,000-to-100,000 extension

After inspecting the completed main matrix, the sole censored run,
`sub_frac0p25_wd1_seed1`, was resumed from its epoch-50,000 optimizer and scheduler
state to epoch 100,000. This was an exploratory follow-up, not a modification of the
approved 27-run study. It lives under `runs/posthoc/`, carries the machine-readable
status `POST_HOC_DO_NOT_POOL_WITH_MAIN_STUDY`, and must not be pooled into the main
results above.

The run did not grok by epoch 100,000. Training accuracy was 1.000 at the endpoint;
test accuracy rose from 0.028923 at epoch 50,000 to 0.102120 at epoch 100,000. Its
highest saved test accuracy was 0.102433 at epoch 99,800, still far below the frozen
0.90 threshold. The trajectory therefore shows gradual improvement over this horizon,
not a behavioral grokking transition.

The isolated artifact contains 501 checkpoints at 100-epoch resolution (the copied
source checkpoint plus 500 continuation checkpoints), totaling 456,951,579 bytes.
Every checkpoint passed a fresh SHA-256 manifest verification. Pre/post hashes of the
source protocol, metrics, checkpoint manifest, latest state, and aggregate main-study
summary are identical. Flow extraction was not run.

The detailed post-hoc report, metrics, provenance, and checkpoint manifest are in
`runs/posthoc/sub_frac0p25_wd1_seed1_to100000/`.

## 10. Purely post-hoc 100,000-to-200,000 extension

At the user's request, the same trajectory was resumed again from the isolated
epoch-100,000 optimizer and scheduler state. The second extension is a new artifact
whose protocol hashes and references the completed 100,000-epoch artifact. It does
not alter the original run, the first extension, or the main-study outcome.

Under the frozen first-crossing definition, the run grokked at epoch 105,000, with
training accuracy 1.000000 and test accuracy 0.918242. Test accuracy was 0.832307 at
104,900, rose to 0.974313 at 105,100, and first reached 1.000000 at epoch 105,400.
Both training and test accuracy were 1.000000 at the epoch-200,000 endpoint.

The post-transition trajectory was not monotonic. Of the 951 saved checkpoints at or
after the first crossing, 15 fell below the joint 0.99/0.90 criterion. These were
isolated one-checkpoint excursions followed by recovery at the next 100-epoch sample;
the lowest occurred at epoch 131,000 (training 0.651942, test 0.639658), and the last
occurred at epoch 196,300. Thus “grokked at 105,000” means the preregistered first
crossing, not permanent threshold retention.

The isolated 100,000-to-200,000 artifact contains 1,001 checkpoints at 100-epoch
resolution, totaling 912,991,079 bytes. All checkpoint hashes passed verification,
and the recorded pre/post hashes of the 100,000-epoch source artifact and aggregate
main-study summary are identical. Flow extraction was not run. Detailed results are
in `runs/posthoc/sub_frac0p25_wd1_seed1_to200000/`.

## 11. Lossless raw-flow extraction optimization

The raw extractor was optimized before the full flow phase. The numerical
decomposition, node catalog, edge aggregation, canonical-path ordering, and record
schema are unchanged. In particular, every successful or degenerate record retains an
explicit `split` value of `train` or `test`.

The optimized implementation performs one model forward pass per example and reuses
its immutable cache across the four flow kinds. The original implementation performed
five forward passes per example: one to choose the competitor and one inside each flow
construction. Independent checkpoints can now be processed by multiple CPU processes;
each worker reuses one model instance and is restricted to one PyTorch thread to avoid
oversubscription. Output files use deterministic gzip headers, are written atomically,
and are recorded in an incrementally updated SHA-256 manifest. Interrupted extractions
can resume only when the existing provenance manifest exactly matches the request.
Every source checkpoint is digest-checked before it is loaded.

Regression tests establish exact equality between cached and uncached `RawFlow`
objects and byte-for-byte equality between serial and parallel compressed artifacts.
A direct comparison against the pre-optimization Git version also produced exact
edge/path equality for all 16 checked flows from four train/test examples at a real
epoch-11,700 checkpoint.
A real 128-width, 512-MLP benchmark measured approximately 33 records/second with one
optimized CPU worker, versus 22 records/second previously. Twelve workers processed 60
checkpoints at approximately 256 records/second including process startup. At that
observed rate, the proposed 876,096-record main matrix would take roughly 57 minutes.
Benchmark outputs were temporary and deleted. No full raw-flow extraction or numbered
definition calculation was run as part of this optimization.
