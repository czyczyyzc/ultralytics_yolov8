# Native Dist Public-Code Adapter

`tracker.cpp` is a behavioral C++ port of the executed axis-aligned, no-ReID,
no-score-fusion path in the public Dist-Tracker repository at commit
`396c359e1aa8be4fd5e81a02626cb1ee3867cf7c`:
https://github.com/earth-insights/Dist-Tracker

The original tracker modules bear the Ultralytics AGPL-3.0 notice; this port
retains that license. It is not the older simplified RK-BoT-SORT and does not
claim paper-specific FLIT/L2-IoU features absent from the executed public code.
XYWH Kalman state/covariance, three association stages, unconfirmed/lost/removed
lifetimes, threshold inclusivity, duplicate removal and output ordering are
preserved. Only current-frame detector observations are emitted by the adapter.

`third_party/lap/lapjv.cpp`, `lapjv.h`, and `LICENSE` are unmodified imports from
https://github.com/gatagat/lap at tag v0.5.12, commit
`600c210d9bef793ee0fe502cbc350e676a6e083a` (BSD-2-Clause license).
Finite cost-limit padding matches its Python wrapper, preserving solver ties.
Do not substitute a different assignment implementation without revalidation.

Build using `bash build.sh OUTPUT_DIRECTORY`. Tracker needs only a C++17 compiler;
native GMC additionally needs OpenCV development headers/libraries. No PyTorch.
No fast-math or fused multiply-add contraction is enabled. Numerical state values
need not be bit-identical across BLAS/C++ implementations; ID/observation
equivalence and numerical state tolerances must be checked separately.
