#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace rk_yolo {
struct DflLayout {
    int bins;
    int first_bin;
};

inline DflLayout dfl_output_layout(int channels, bool signed_p3_location) {
    if (channels == 64) return {16, 0};
    if (channels == 68 && signed_p3_location) return {17, -1};
    throw std::runtime_error("Unsupported RKNN DFL output layout; check the signed P3 export contract");
}

template <typename ReadChannel>
float dfl_expectation(DflLayout layout, int side, ReadChannel read_channel) {
    std::array<float, 17> logits;
    for (int bin = 0; bin < layout.bins; ++bin) {
        logits[bin] = read_channel(side * layout.bins + bin);
    }
    const float max_logit = *std::max_element(logits.begin(), logits.begin() + layout.bins);
    float sum = 0.0f;
    float weighted = 0.0f;
    for (int bin = 0; bin < layout.bins; ++bin) {
        const float value = std::exp(logits[bin] - max_logit);
        sum += value;
        weighted += value * static_cast<float>(layout.first_bin + bin);
    }
    return weighted / std::max(sum, 1e-12f);
}
}  // namespace rk_yolo
