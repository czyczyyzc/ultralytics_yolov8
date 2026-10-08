// SPDX-License-Identifier: AGPL-3.0-only
// Configuration ABI shared by the offline adapter and the native video pipeline.
#pragma once
#include <array>

inline std::array<double,14> motion_defaults(double fps) {
    // high, low, birth, expiry seconds, fps, localization floor, acceleration,
    // unknown-camera speed, innovation speed gate, radius cap, NIS gate, hits, window, ambiguity.
    return {.03,.01,.10,1.,fps,1.5,180.,1500.,3000.,240.,16.,3.,4.,.03};
}
