#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "detector_based_tracker.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: detector_based_tracker_replay DETECTIONS.csv FRAME_COUNT FPS [FIRST_COST SECOND_COST]\n";
        return 1;
    }
    const int frame_count = std::stoi(argv[2]);
    const double fps = std::stod(argv[3]);
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error("Unable to open detections CSV");

    std::unordered_map<int, std::vector<rk_tracker::Detection>> detections_by_frame;
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        std::stringstream stream(line);
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(stream, field, ',')) fields.push_back(field);
        if (fields.size() < 7) continue;
        const int frame = std::stoi(fields[0]);
        const float x = std::stof(fields[1]);
        const float y = std::stof(fields[2]);
        const float width = std::stof(fields[3]);
        const float height = std::stof(fields[4]);
        detections_by_frame[frame].push_back(rk_tracker::Detection{
            x, y, x + width, y + height, std::stof(fields[5]), std::stoi(fields[6]),
        });
    }

    rk_tracker::Config config;
    config.fallback_fps = fps;
    if (argc >= 6) {
        config.first_match_cost = std::stof(argv[4]);
        config.second_match_cost = std::stof(argv[5]);
    }
    rk_tracker::DetectorBasedTracker tracker(config);
    std::set<int> ids;
    int predicted_outputs = 0;
    int output_frames = 0;
    int last_observed_id = -1;
    for (int frame = 0; frame < frame_count; ++frame) {
        const auto outputs = tracker.update(detections_by_frame[frame], frame / fps, 1920, 1080);
        if (!outputs.empty()) ++output_frames;
        for (const auto& output : outputs) {
            ids.insert(output.track_id);
            if (output.predicted) ++predicted_outputs;
            if (!output.predicted && output.track_id != last_observed_id) {
                std::cout << "observed_id_change frame=" << frame << " id=" << output.track_id << '\n';
                last_observed_id = output.track_id;
            }
        }
    }
    std::cout << "unique_ids=" << ids.size()
              << " output_frames=" << output_frames
              << " predicted_outputs=" << predicted_outputs << '\n';
    return 0;
}
