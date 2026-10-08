// SPDX-License-Identifier: AGPL-3.0-only
// Configuration ABI shared by the offline adapter and the native video pipeline.
#pragma once
#include <array>

// All current detections are retained; only Confirmed observations expose an ID.
// motion_observations(handle, triples, capacity): [id_or_zero, detection_index, status].
enum class MotionObservationStatus { Unassigned=0, Pending=1, Confirmed=2, Ambiguous=3, BelowLow=4 };

inline std::array<double,14> motion_defaults(double fps) {
    // high, low, birth, expiry seconds, fps, localization floor, acceleration,
    // unknown-camera speed, innovation speed gate, radius cap, NIS gate, hits, window, ambiguity.
    return {.03,.01,.10,1.,fps,1.5,180.,1500.,3000.,240.,16.,3.,4.,.03};
}
