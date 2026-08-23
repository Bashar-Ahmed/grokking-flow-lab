# MATS application visual addendum

This directory is a reviewer-facing summary generated entirely from the completed
main study and explicitly isolated post-hoc analysis. Lower-level reports and raw
artifacts remain the source of record.

## Recommended core sequence

1. [Experiment coverage](figure01_experiment_coverage.png): 26/27 runs grokked
   by 50k; the dot plot exposes seed and setting variability.
2. [Definition consistency](figure02_definition_consistency.png): Definition-03 is
   the only candidate increasing for every evaluable run in every operation and split
   over the fixed pre/post windows.
3. [Forecast calibration](figure03_forecast_calibration.png): actual-versus-predicted
   timing makes forecast gains and large residual errors visible.
4. [Train/test agreement](figure04_train_test_agreement.png): Definition-03 changes
   agree across example splits (correlation 0.959).

## Robustness and research judgment

5. [Forecast-window sensitivity](figure05_forecast_window_sensitivity.png): the
   Safe-Mass result depends on using four rather than two or three checkpoints.
6. [Definition-selection stability](figure06_definition_selection_stability.png):
   Definition-04 dominates seed/setting selection, while task transfer is unstable.
7. [Censored-run continuation](figure07_censored_run_posthoc.png): the sole censored
   main run groks at epoch 105,000, but only in isolated post-hoc analysis.
8. [Research pipeline](figure08_research_pipeline.png): concise scope and provenance.

Every result figure has a PDF counterpart and a CSV containing its plotted values.
The pipeline schematic has no underlying numeric table. `MANIFEST.json` records
SHA-256 and byte size for every addendum file except itself.

## Interpretation boundary

These figures support a replicated structural transition and a modest, window-sensitive
within-setting forecast signal. They do not establish causal mechanism, universal
operation transfer, or a deployable online predictor. The application should use the
four core figures in the main work sample and keep Figures 5–8 as supporting evidence.
