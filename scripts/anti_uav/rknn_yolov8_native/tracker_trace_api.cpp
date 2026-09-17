// Diagnostic build only; the production C API and board binary are unchanged.
#define RK_TRACKER_DIAGNOSTICS
#include "tracker_c_api.cpp"

#ifndef RK_TRACKER_LEGACY_TRACE
extern "C" int rk_tracker_set_confirmed_first(void* tracker, int enabled) {
    if (!tracker) return -1;
    static_cast<rk_tracker::DetectorBasedTracker*>(tracker)->diagnostic_confirmed_first(enabled != 0);
    return 0;
}

extern "C" int rk_tracker_set_active_first(void* tracker, int enabled) {
    if (!tracker) return -1;
    static_cast<rk_tracker::DetectorBasedTracker*>(tracker)->diagnostic_active_first(enabled != 0);
    return 0;
}
#endif

extern "C" int rk_tracker_assign(const double* costs, int rows, int cols,
                                  double threshold, int* output) {
    if (rows < 0 || cols < 0) return -1;
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) matrix[row][col] = costs[row * cols + col];
    }
    const auto result = rk_tracker::detail::hungarian_assignment(matrix, threshold);
    std::copy(result.begin(), result.end(), output);
    return static_cast<int>(result.size());
}

extern "C" int rk_tracker_trace(void* tracker, const float* boxes, int count,
                                 double timestamp, double* output, int capacity) {
    try {
        if (!tracker || count < 0 || capacity < 0) return -1;
        std::vector<rk_tracker::Detection> detections;
        for (int i = 0; i < count; ++i) {
            const float* b = boxes + i * 5;
            detections.push_back({b[0], b[1], b[2], b[3], b[4], 0});
        }
        const auto rows = static_cast<rk_tracker::DetectorBasedTracker*>(tracker)->diagnostic_costs(detections, timestamp);
        if (static_cast<int>(rows.size()) > capacity) {
            last_error = "Trace output capacity exceeded";
            return -1;
        }
        for (size_t i = 0; i < rows.size(); ++i) {
            std::copy(rows[i].begin(), rows[i].end(), output + i * 15);
        }
        return static_cast<int>(rows.size());
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}
