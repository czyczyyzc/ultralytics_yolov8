# Launch Record

Started on server 47.107.185.207 at 2026-09-16 17:07:14 +0800.
Parent PID: 1615969. Launch commit: `ebaa36c`.

Run:
`/mnt/chenziye/codes/ultralytics_yolov8/runs/anti_uav/native_fullpool_14_vs_28_20260916`

At initial verification, the parent and extraction child were alive, the stage
was `prepare_expanded`, and the local validation font was present. Model training
had not yet started: raw video/annotation checksums and full-frame extraction run
first. The detached pipeline automatically trains and evaluates after preparation
succeeds. On an error it records `failed` rather than silently skipping a video.

Current status must be read from `status.json`, not inferred from this dated record.
Logs: `pipeline.log`, `logs/prepare_expanded.log`, then
`logs/train_expanded_28.log` and `logs/train_baseline_14.log`.

Twenty-two server tests passed, including two-worker finite training-loader
cycling, label-pool coverage, held-out-frame rejection and fixed 544x960 validation
with a native-coordinate ground-truth round trip. Local code was pushed to origin
and then fast-forward pulled on the server using a Git bundle.

No old process was killed for this launch and no deployment model was replaced.
GPU 6 retains its existing approximately 0.6GB Triton service. New training will
use the original photometric/geometric recipe, not the discontinued extra crops.

## Revised Sampling Decision

The initial proposal of positive stride 3 / negative stride 20 was superseded
before any new extraction or training was launched. Both extraction strides are
now 1. The complete new reviewed pool is 55.31% positive and 44.69% negative.
Training uses a global approximately 85% / 15% ratio while preserving every old
slot, all new positives each epoch, and cycling all new negatives within five
epochs. The 101 uncertain frames are excluded by manifest policy; uncertainty
is not inferred from or converted into an empty YOLO label.
