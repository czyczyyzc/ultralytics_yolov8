#include <cassert>
#include <cmath>
#include <iostream>
#include <functional>
#include <random>
#include <vector>

#include "detector_based_tracker.hpp"

namespace {

rk_tracker::Detection detection(float center_x, float center_y, float score = 0.9f) {
    return rk_tracker::Detection{
        center_x - 10.0f, center_y - 6.0f,
        center_x + 10.0f, center_y + 6.0f,
        score, 0,
    };
}

void test_constant_velocity_and_dropout() {
    rk_tracker::Config config;
    config.fallback_fps = 107.0;
    config.min_hits = 2;
    config.prediction_grace_sec = 0.20;
    rk_tracker::DetectorBasedTracker tracker(config);

    int stable_id = -1;
    int predicted_frames = 0;
    for (int frame = 0; frame < 80; ++frame) {
        const double timestamp = frame / 107.0;
        std::vector<rk_tracker::Detection> detections;
        if (frame < 35 || frame > 41) detections.push_back(detection(100.0f + frame * 1.5f, 80.0f));
        const std::vector<rk_tracker::TrackOutput> outputs = tracker.update(detections, timestamp, 640, 480);
        assert(outputs.size() == 1);
        if (stable_id < 0) stable_id = outputs.front().track_id;
        assert(outputs.front().track_id == stable_id);
        if (frame >= 35 && frame <= 41) {
            assert(outputs.front().predicted);
            ++predicted_frames;
        }
    }
    assert(predicted_frames == 7);
}

void test_low_confidence_second_association() {
    rk_tracker::DetectorBasedTracker tracker;
    int stable_id = -1;
    for (int frame = 0; frame < 12; ++frame) {
        const float score = frame == 8 ? 0.20f : 0.90f;
        const auto outputs = tracker.update({detection(200.0f + frame, 120.0f, score)}, frame / 30.0, 640, 480);
        assert(outputs.size() == 1);
        if (stable_id < 0) stable_id = outputs.front().track_id;
        assert(outputs.front().track_id == stable_id);
        if (frame == 8) assert(!outputs.front().predicted);
    }
}

void test_two_targets_keep_distinct_ids() {
    rk_tracker::DetectorBasedTracker tracker;
    int first_id = -1;
    int second_id = -1;
    for (int frame = 0; frame < 40; ++frame) {
        const float left_to_right = 80.0f + frame * 3.0f;
        const float right_to_left = 360.0f - frame * 3.0f;
        const auto outputs = tracker.update(
            {detection(left_to_right, 100.0f), detection(right_to_left, 180.0f)},
            frame / 60.0, 640, 480);
        assert(outputs.size() == 2);
        if (frame == 0) {
            first_id = outputs[0].track_id;
            second_id = outputs[1].track_id;
            assert(first_id != second_id);
        }
        assert(outputs[0].track_id == first_id);
        assert(outputs[1].track_id == second_id);
    }
}

void test_non_monotonic_timestamp_fallback() {
    rk_tracker::DetectorBasedTracker tracker;
    const auto first = tracker.update({detection(50.0f, 50.0f)}, 1.0);
    const auto second = tracker.update({detection(51.0f, 50.0f)}, 1.0);
    assert(first.size() == 1 && second.size() == 1);
    assert(first.front().track_id == second.front().track_id);
    assert(std::isfinite(second.front().x1));
}

void test_camera_motion_hook() {
    rk_tracker::Config config;
    config.min_hits = 1;
    config.prediction_grace_sec = 0.5;
    rk_tracker::DetectorBasedTracker tracker(config);
    const auto first = tracker.update({detection(50.0f, 50.0f)}, 0.0);
    rk_tracker::CameraMotion motion;
    motion.a02 = 25.0;
    motion.a12 = -10.0;
    const auto shifted = tracker.update({}, 0.1, 640, 480, motion);
    assert(first.size() == 1 && shifted.size() == 1);
    assert(std::abs(shifted.front().x1 - (first.front().x1 + 25.0f)) < 0.1f);
    assert(std::abs(shifted.front().y1 - (first.front().y1 - 10.0f)) < 0.1f);
}

void test_assignment_rejects_invalid_edges_before_optimization() {
    const auto result = rk_tracker::detail::hungarian_assignment({{0.10, 0.93}, {0.20, 10.0}}, 0.92);
    assert((result == std::vector<int>{0, -1}));
}

void test_assignment_boundary_and_nonfinite_costs() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    assert((rk_tracker::detail::hungarian_assignment({{0.92, nan, inf, -inf}}, 0.92) ==
            std::vector<int>{-1}));
    assert((rk_tracker::detail::hungarian_assignment({{nan, 0.2}, {0.1, inf}}, 0.92) ==
            std::vector<int>{1, 0}));
    assert(rk_tracker::detail::hungarian_assignment({}, 0.92).empty());
    assert((rk_tracker::detail::hungarian_assignment({{}, {}}, 0.92) == std::vector<int>{-1, -1}));
}

void test_expiry_before_association() {
    rk_tracker::Config config;
    config.min_hits = 1;
    config.track_buffer_sec = 1.0;
    rk_tracker::DetectorBasedTracker tracker(config);
    const auto first = tracker.update({detection(100.0f, 100.0f)}, 0.0);
    const auto within = tracker.update({detection(100.0f, 100.0f)}, 0.5);
    assert(within.front().track_id == first.front().track_id);
    const auto boundary = tracker.update({detection(100.0f, 100.0f)}, 1.5);
    assert(boundary.front().track_id == first.front().track_id);
    const auto expired = tracker.update({detection(100.0f, 100.0f)}, 2.51);
    assert(expired.size() == 1);
    assert(expired.front().track_id != first.front().track_id);
    assert(tracker.active_track_count() == 1);
    tracker.update({}, 4.0);
    assert(tracker.active_track_count() == 0);
}

void test_opt_in_confirmed_priority() {
    for (bool priority : {false, true}) {
        rk_tracker::Config config;
        config.min_hits = 3;
        config.confirmed_first = priority;
        rk_tracker::DetectorBasedTracker tracker(config);
        int stable_id = -1;
        for (int frame = 0; frame < 3; ++frame) {
            stable_id = tracker.update({detection(100, 100)}, frame / 100.0).front().track_id;
        }
        const auto split = tracker.update({detection(100, 100), detection(112, 100)}, 0.03);
        assert(split.size() == 2);
        const auto next = tracker.update({detection(112, 100)}, 0.04);
        assert(next.size() == 1);
        assert((next.front().track_id == stable_id) == priority);
        assert(next.front().confirmed == priority);
    }
}

void test_assignment_matches_brute_force_objective() {
    std::mt19937 generator(20260917);
    std::uniform_real_distribution<double> random_cost(0.0, 1.3);
    constexpr double threshold = 0.92;
    for (int rows = 1; rows <= 3; ++rows) {
        for (int cols = 1; cols <= 3; ++cols) {
            for (int sample = 0; sample < 100; ++sample) {
                std::vector<std::vector<double>> costs(rows, std::vector<double>(cols));
                for (auto& row : costs) for (double& cost : row) cost = random_cost(generator);
                double best = std::numeric_limits<double>::infinity();
                std::function<void(int, unsigned, double, int)> search = [&](int row, unsigned used, double cost, int matches) {
                    if (row == rows) {
                        best = std::min(best, cost + (cols - matches) * threshold);
                        return;
                    }
                    search(row + 1, used, cost + threshold, matches);
                    for (int col = 0; col < cols; ++col) {
                        if (!(used & (1U << col)) && costs[row][col] < threshold) {
                            search(row + 1, used | (1U << col), cost + costs[row][col], matches + 1);
                        }
                    }
                };
                search(0, 0, 0.0, 0);
                const auto assignment = rk_tracker::detail::hungarian_assignment(costs, threshold);
                double actual = 0.0;
                unsigned used = 0;
                int matches = 0;
                for (int row = 0; row < rows; ++row) {
                    const int col = assignment[row];
                    if (col < 0) { actual += threshold; continue; }
                    assert(!(used & (1U << col)));
                    assert(costs[row][col] < threshold);
                    used |= 1U << col;
                    ++matches;
                    actual += costs[row][col];
                }
                actual += (cols - matches) * threshold;
                assert(std::abs(actual - best) < 1e-9);
            }
        }
    }
}

}  // namespace

int main() {
    test_constant_velocity_and_dropout();
    test_low_confidence_second_association();
    test_two_targets_keep_distinct_ids();
    test_non_monotonic_timestamp_fallback();
    test_camera_motion_hook();
    test_assignment_rejects_invalid_edges_before_optimization();
    test_assignment_boundary_and_nonfinite_costs();
    test_expiry_before_association();
    test_opt_in_confirmed_priority();
    test_assignment_matches_brute_force_objective();
    std::cout << "detector_based_tracker tests passed\n";
    return 0;
}
