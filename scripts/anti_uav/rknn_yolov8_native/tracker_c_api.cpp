#include "detector_based_tracker.hpp"
#include <exception>
#include <string>

namespace { thread_local std::string last_error; }

extern "C" {
const char* rk_tracker_error() { return last_error.c_str(); }

void* rk_tracker_create(double high, double low, double birth, double first,
                        double second, double buffer, double grace, double fps,
                        int min_hits) {
    try {
        rk_tracker::Config config;
        config.high_threshold = high;
        config.low_threshold = low;
        config.new_track_threshold = birth;
        config.first_match_cost = first;
        config.second_match_cost = second;
        config.track_buffer_sec = buffer;
        config.prediction_grace_sec = grace;
        config.fallback_fps = fps;
        config.min_hits = min_hits;
        return new rk_tracker::DetectorBasedTracker(config);
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

void rk_tracker_destroy(void* tracker) {
    delete static_cast<rk_tracker::DetectorBasedTracker*>(tracker);
}

// Input: Nx5 xyxy,score. Output: Nx12 id,xyxy,score,confirmed,predicted,
// age,hits,seconds_since_update,class. This bridge shares the board tracker.
int rk_tracker_update(void* tracker, const float* boxes, int count, double timestamp,
                      int width, int height, double* output, int capacity) {
    try {
        if (!tracker || count < 0 || capacity < 0) return -1;
        std::vector<rk_tracker::Detection> detections;
        detections.reserve(count);
        for (int i = 0; i < count; ++i) {
            const float* box = boxes + i * 5;
            detections.push_back({box[0], box[1], box[2], box[3], box[4], 0});
        }
        auto tracks = static_cast<rk_tracker::DetectorBasedTracker*>(tracker)->update(
            detections, timestamp, width, height);
        if (static_cast<int>(tracks.size()) > capacity) {
            last_error = "Track output capacity exceeded";
            return -1;
        }
        for (size_t i = 0; i < tracks.size(); ++i) {
            const auto& t = tracks[i];
            double* row = output + i * 12;
            row[0] = t.track_id;
            row[1] = t.x1; row[2] = t.y1; row[3] = t.x2; row[4] = t.y2;
            row[5] = t.score; row[6] = t.confirmed; row[7] = t.predicted;
            row[8] = t.age; row[9] = t.hits;
            row[10] = t.time_since_update_sec; row[11] = t.class_id;
        }
        return static_cast<int>(tracks.size());
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}
}
