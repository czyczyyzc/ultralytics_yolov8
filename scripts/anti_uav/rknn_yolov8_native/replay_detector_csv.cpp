#include "detector_based_tracker.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
    if (argc != 7 && argc != 15) {
        std::cerr << "Usage: replay_detector_csv DETECTIONS.csv TRACKS.csv FRAME_COUNT FPS WIDTH HEIGHT "
                     "[HIGH LOW NEW FIRST_COST SECOND_COST BUFFER_SEC PREDICTION_SEC MIN_HITS]\n";
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
    const auto header = split_csv(line);
    std::unordered_map<std::string, size_t> columns;
    for (size_t index = 0; index < header.size(); ++index) columns[header[index]] = index;
    for (const char* required : {"frame", "x", "y", "width", "height", "confidence", "class_id"}) {
        if (!columns.count(required)) throw std::runtime_error(std::string("Missing CSV column: ") + required);
    }
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split_csv(line);
        if (fields.size() < header.size()) throw std::runtime_error("Malformed detector CSV row");
        const int frame = std::stoi(fields[columns.at("frame")]);
        if (frame < 0 || frame >= frame_count) throw std::runtime_error("Detector frame out of range");
        const float x = std::stof(fields[columns.at("x")]);
        const float y = std::stof(fields[columns.at("y")]);
        const float box_width = std::stof(fields[columns.at("width")]);
        const float box_height = std::stof(fields[columns.at("height")]);
        frames[frame].push_back(
            {x, y, x + box_width, y + box_height,
             std::stof(fields[columns.at("confidence")]), std::stoi(fields[columns.at("class_id")])});
    }

    rk_tracker::Config config;
    config.high_threshold = 0.03f;
    config.low_threshold = 0.01f;
    config.new_track_threshold = 0.05f;
    config.first_match_cost = 0.92f;
    config.second_match_cost = 0.92f;
    config.track_buffer_sec = 1.0;
    config.prediction_grace_sec = 0.0;
    config.fallback_fps = fps;
    config.min_hits = 2;
    if (argc == 15) {
        config.high_threshold = std::stof(argv[7]);
        config.low_threshold = std::stof(argv[8]);
        config.new_track_threshold = std::stof(argv[9]);
        config.first_match_cost = std::stof(argv[10]);
        config.second_match_cost = std::stof(argv[11]);
        config.track_buffer_sec = std::stod(argv[12]);
        config.prediction_grace_sec = std::stod(argv[13]);
        config.min_hits = std::stoi(argv[14]);
    }
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
