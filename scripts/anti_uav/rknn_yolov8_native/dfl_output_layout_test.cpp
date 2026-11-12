#include "dfl_output_layout.hpp"
#include <cassert>

int main() {
    const auto signed_p3 = rk_yolo::dfl_output_layout(68, true);
    const auto legacy = rk_yolo::dfl_output_layout(64, false);
    assert(std::abs(rk_yolo::dfl_expectation(legacy, 0, [](int) { return 0.f; }) - 7.5f) < 1e-6f);
    const float left = rk_yolo::dfl_expectation(signed_p3, 0, [](int c) {
        if (c == 0) return std::log(.125f);
        if (c == 1) return std::log(.875f);
        return -30.f;
    });
    const float right = rk_yolo::dfl_expectation(signed_p3, 2, [](int c) {
        return (c == 35 || c == 36) ? std::log(.5f) : -30.f;
    });
    assert(std::abs((.5f-left)*8.f-5.f) < 1e-6f);
    assert(std::abs((.5f+right)*8.f-8.f) < 1e-6f);
    try {
        rk_yolo::dfl_output_layout(68, false);
        assert(false);
    } catch (const std::runtime_error&) {}
}
