#include <iostream>
#include <string>
#include "detector_based_tracker.hpp"

int main() {
    int failures = 0;
    const auto assignment = rk_tracker::detail::hungarian_assignment({{0.10, 0.93}, {0.20, 10.0}}, 0.92);
    const bool gated = assignment.size() == 2 && assignment[0] == 0 && assignment[1] == -1;
    std::cout << "invalid_edge_cannot_steal_valid_assignment=" << gated
              << " actual=" << assignment[0] << "," << assignment[1] << '\n';
    failures += !gated;

    rk_tracker::Config config;
    config.min_hits = 1;
    config.track_buffer_sec = 1.0;
    rk_tracker::DetectorBasedTracker tracker(config);
    const rk_tracker::Detection box{100, 100, 110, 110, .9f, 0};
    const auto first = tracker.update({box}, 0.0, 640, 480);
    const auto expired = tracker.update({box}, 1.5, 640, 480);
    const bool not_revived = expired.size() == 1 && expired[0].track_id != first[0].track_id;
    std::cout << "expired_track_cannot_revive=" << not_revived << '\n';
    failures += !not_revived;
    return failures ? 1 : 0;
}
