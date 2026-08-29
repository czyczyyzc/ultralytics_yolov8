#include "detector_based_tracker.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    return fields;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: replay_detector_csv DETECTIONS.csv TRACKS.csv FRAME_COUNT FPS WIDTH HEIGHT\n";
        return 2;
    }
    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    const int frame_count = std::stoi(argv[3]);
    const double fps = std::stod(argv[4]);
    const int width = std::stoi(argv[5]);
    const int height = std::stoi(argv[6]);
    if (frame_count <= 0 || fps <= 0.0) throw std::runtime_error("Invalid frame count or FPS");

    std::vector<std::vector<rk_tracker::Detection>> frames(frame_count);
    std::ifstream input(input_path);
    if (!input) throw std::runtime_error("Unable to open detector CSV");
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split_csv(line);
        if (fields.size() < 9) throw std::runtime_error("Malformed detector CSV row");
        const int frame = std::stoi(fields[0]);
        if (frame < 0 || frame >= frame_count) throw std::runtime_error("Detector frame out of range");
        const float x = std::stof(fields[3]);
        const float y = std::stof(fields[4]);
        const float box_width = std::stof(fields[5]);
        const float box_height = std::stof(fields[6]);
        frames[frame].push_back(
            {x, y, x + box_width, y + box_height, std::stof(fields[7]), std::stoi(fields[8])});
    }

    rk_tracker::Config config;
    config.high_threshold = 0.45f;
    config.low_threshold = 0.10f;
    config.new_track_threshold = 0.50f;
    config.first_match_cost = 0.92f;
    config.second_match_cost = 0.92f;
    config.track_buffer_sec = 1.0;
    config.prediction_grace_sec = 0.0;
    config.fallback_fps = fps;
    config.min_hits = 2;
    rk_tracker::DetectorBasedTracker tracker(config);

    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("Unable to open track CSV");
    output << "frame,timestamp_sec,track_id,x,y,width,height,confidence,class_id,predicted,confirmed,age,hits,time_since_update_sec\n";
    output << std::fixed << std::setprecision(6);
    for (int frame = 0; frame < frame_count; ++frame) {
        const double timestamp = frame / fps;
        const auto tracks = tracker.update(frames[frame], timestamp, width, height);
        for (const auto& track : tracks) {
            output << frame << ',' << timestamp << ',' << track.track_id << ','
                   << track.x1 << ',' << track.y1 << ',' << track.x2 - track.x1 << ','
                   << track.y2 - track.y1 << ',' << track.score << ',' << track.class_id << ','
                   << static_cast<int>(track.predicted) << ',' << static_cast<int>(track.confirmed) << ','
                   << track.age << ',' << track.hits << ',' << track.time_since_update_sec << '\n';
        }
    }
    return 0;
}
