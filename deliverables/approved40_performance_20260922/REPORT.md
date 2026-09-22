# Add-on branch calibration

Validation calibration, not independent test or RKNN/tracker results

Thresholds are selected on Video00009 validation only. No independent-test improvement is claimed.

| Policy | P3 threshold | P2 threshold | P | R | FP | FN | 4-8px R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.03 | 0.03 | 66.19% | 75.91% | 309 | 192 | 77.81% |
| selected_recall_safe | 0.075 | 0.03 | 67.60% | 75.91% | 290 | 192 | 77.81% |
| best_at_same_fp_budget | 0.25 | 0.02 | 66.23% | 76.04% | 309 | 191 | 78.06% |
