#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace rk_tracker {

struct Detection {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    int class_id = 0;
};

struct CameraMotion {
    // Affine transform from the previous image coordinates to the current frame.
    double a00 = 1.0;
    double a01 = 0.0;
    double a02 = 0.0;
    double a10 = 0.0;
    double a11 = 1.0;
    double a12 = 0.0;

    bool is_identity() const {
        return a00 == 1.0 && a01 == 0.0 && a02 == 0.0 &&
               a10 == 0.0 && a11 == 1.0 && a12 == 0.0;
    }
};

struct TrackOutput {
    int track_id = 0;
    int class_id = 0;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    bool predicted = false;
    bool confirmed = false;
    int age = 0;
    int hits = 0;
    double time_since_update_sec = 0.0;
};

struct Config {
    float high_threshold = 0.45f;
    float low_threshold = 0.10f;
    float new_track_threshold = 0.50f;
    float first_match_cost = 0.92f;
    float second_match_cost = 0.92f;
    double track_buffer_sec = 1.0;
    double prediction_grace_sec = 0.0;
    double fallback_fps = 107.0;
    int min_hits = 2;
};

namespace detail {

inline double box_iou(const std::array<double, 4>& lhs, const Detection& rhs) {
    const double x1 = std::max(lhs[0], static_cast<double>(rhs.x1));
    const double y1 = std::max(lhs[1], static_cast<double>(rhs.y1));
    const double x2 = std::min(lhs[2], static_cast<double>(rhs.x2));
    const double y2 = std::min(lhs[3], static_cast<double>(rhs.y2));
    const double intersection = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    const double lhs_area = std::max(0.0, lhs[2] - lhs[0]) * std::max(0.0, lhs[3] - lhs[1]);
    const double rhs_area = std::max(0.0f, rhs.x2 - rhs.x1) * std::max(0.0f, rhs.y2 - rhs.y1);
    return intersection / std::max(lhs_area + rhs_area - intersection, 1e-9);
}

inline std::vector<int> hungarian_assignment(
    const std::vector<std::vector<double>>& costs,
    double unmatched_cost) {
    const int rows = static_cast<int>(costs.size());
    const int cols = rows == 0 ? 0 : static_cast<int>(costs.front().size());
    if (rows == 0) return {};
    if (cols == 0) return std::vector<int>(rows, -1);

    // Dedicated dummy rows and columns allow both tracks and detections to stay unmatched.
    const int size = rows + cols;
    const double invalid_cost = unmatched_cost + 1000.0;
    std::vector<std::vector<double>> square(size, std::vector<double>(size, invalid_cost));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) square[row][col] = costs[row][col];
        square[row][cols + row] = unmatched_cost;
    }
    for (int col = 0; col < cols; ++col) {
        square[rows + col][col] = unmatched_cost;
        for (int dummy_col = cols; dummy_col < size; ++dummy_col) {
            square[rows + col][dummy_col] = 0.0;
        }
    }

    std::vector<double> u(size + 1, 0.0), v(size + 1, 0.0);
    std::vector<int> p(size + 1, 0), way(size + 1, 0);
    for (int row = 1; row <= size; ++row) {
        p[0] = row;
        int col0 = 0;
        std::vector<double> min_value(size + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(size + 1, false);
        do {
            used[col0] = true;
            const int current_row = p[col0];
            double delta = std::numeric_limits<double>::infinity();
            int col1 = 0;
            for (int col = 1; col <= size; ++col) {
                if (used[col]) continue;
                const double current = square[current_row - 1][col - 1] - u[current_row] - v[col];
                if (current < min_value[col]) {
                    min_value[col] = current;
                    way[col] = col0;
                }
                if (min_value[col] < delta) {
                    delta = min_value[col];
                    col1 = col;
                }
            }
            for (int col = 0; col <= size; ++col) {
                if (used[col]) {
                    u[p[col]] += delta;
                    v[col] -= delta;
                } else {
                    min_value[col] -= delta;
                }
            }
            col0 = col1;
        } while (p[col0] != 0);
        do {
            const int col1 = way[col0];
            p[col0] = p[col1];
            col0 = col1;
        } while (col0 != 0);
    }

    std::vector<int> assignment(rows, -1);
    for (int col = 1; col <= size; ++col) {
        const int row = p[col] - 1;
        const int real_col = col - 1;
        if (row >= 0 && row < rows && real_col < cols && costs[row][real_col] < unmatched_cost) {
            assignment[row] = real_col;
        }
    }
    return assignment;
}

class ScalarKalman {
public:
    void initialize(double position, double position_variance, double velocity_variance) {
        position_ = position;
        velocity_ = 0.0;
        p00_ = position_variance;
        p01_ = 0.0;
        p10_ = 0.0;
        p11_ = velocity_variance;
    }

    void predict(double dt, double process_position_variance, double process_velocity_variance) {
        position_ += velocity_ * dt;
        const double old_p00 = p00_;
        const double old_p01 = p01_;
        const double old_p10 = p10_;
        const double old_p11 = p11_;
        p00_ = old_p00 + dt * (old_p01 + old_p10) + dt * dt * old_p11 + process_position_variance;
        p01_ = old_p01 + dt * old_p11;
        p10_ = old_p10 + dt * old_p11;
        p11_ = old_p11 + process_velocity_variance;
    }

    void update(double measurement, double measurement_variance) {
        const double innovation_variance = std::max(p00_ + measurement_variance, 1e-9);
        const double gain_position = p00_ / innovation_variance;
        const double gain_velocity = p10_ / innovation_variance;
        const double innovation = measurement - position_;
        position_ += gain_position * innovation;
        velocity_ += gain_velocity * innovation;

        const double old_p00 = p00_;
        const double old_p01 = p01_;
        const double old_p10 = p10_;
        const double old_p11 = p11_;
        p00_ = (1.0 - gain_position) * old_p00;
        p01_ = (1.0 - gain_position) * old_p01;
        p10_ = old_p10 - gain_velocity * old_p00;
        p11_ = old_p11 - gain_velocity * old_p01;
        p00_ = std::max(p00_, 1e-6);
        p11_ = std::max(p11_, 1e-6);
    }

    double squared_distance(double measurement, double measurement_variance) const {
        const double residual = measurement - position_;
        return residual * residual / std::max(p00_ + measurement_variance, 1e-9);
    }

    double position() const { return position_; }
    double velocity() const { return velocity_; }

    void set_position_velocity(double position, double velocity) {
        position_ = position;
        velocity_ = velocity;
    }

private:
    double position_ = 0.0;
    double velocity_ = 0.0;
    double p00_ = 1.0;
    double p01_ = 0.0;
    double p10_ = 0.0;
    double p11_ = 1.0;
};

}  // namespace detail

class DetectorBasedTracker {
public:
    explicit DetectorBasedTracker(Config config = {}) : config_(config) {}

    std::vector<TrackOutput> update(
        const std::vector<Detection>& detections,
        double timestamp_sec,
        int image_width = 0,
        int image_height = 0,
        const CameraMotion& camera_motion = {}) {
        timestamp_sec = normalize_timestamp(timestamp_sec);
        for (Track& track : tracks_) track.predict(timestamp_sec, camera_motion);

        std::vector<int> high_detections;
        std::vector<int> low_detections;
        for (int index = 0; index < static_cast<int>(detections.size()); ++index) {
            if (detections[index].score >= config_.high_threshold) {
                high_detections.push_back(index);
            } else if (detections[index].score >= config_.low_threshold) {
                low_detections.push_back(index);
            }
        }

        std::vector<int> active_tracks(tracks_.size());
        std::iota(active_tracks.begin(), active_tracks.end(), 0);
        std::vector<bool> matched_track(tracks_.size(), false);
        std::vector<bool> matched_detection(detections.size(), false);

        associate(
            active_tracks, high_detections, detections, config_.first_match_cost,
            matched_track, matched_detection);

        std::vector<int> unmatched_tracks;
        for (int index : active_tracks) {
            if (!matched_track[index]) unmatched_tracks.push_back(index);
        }
        associate(
            unmatched_tracks, low_detections, detections, config_.second_match_cost,
            matched_track, matched_detection);

        for (int index = 0; index < static_cast<int>(tracks_.size()); ++index) {
            if (!matched_track[index]) tracks_[index].mark_missed(timestamp_sec);
        }
        for (int detection_index : high_detections) {
            if (!matched_detection[detection_index] &&
                detections[detection_index].score >= config_.new_track_threshold) {
                tracks_.emplace_back(next_track_id_++, detections[detection_index], timestamp_sec);
            }
        }

        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track& track) {
                return track.time_since_update(timestamp_sec) > config_.track_buffer_sec;
            }),
            tracks_.end());

        std::vector<TrackOutput> outputs;
        outputs.reserve(tracks_.size());
        for (const Track& track : tracks_) {
            const double missed_sec = track.time_since_update(timestamp_sec);
            const bool predicted = missed_sec > 1e-9;
            const bool confirmed = track.hits >= config_.min_hits;
            if (predicted && (!confirmed || missed_sec > config_.prediction_grace_sec)) continue;
            TrackOutput output = track.output(timestamp_sec, predicted, confirmed, image_width, image_height);
            outputs.push_back(output);
        }
        std::sort(outputs.begin(), outputs.end(), [](const TrackOutput& lhs, const TrackOutput& rhs) {
            return lhs.track_id < rhs.track_id;
        });
        return outputs;
    }

    std::size_t active_track_count() const { return tracks_.size(); }

private:
    struct Track {
        int id;
        int class_id;
        float score;
        int age = 1;
        int hits = 1;
        int consecutive_hits = 1;
        double last_update_sec;
        double last_predict_sec;
        Detection last_observation;
        std::array<double, 4> observation_velocity{};
        bool has_observation_velocity = false;
        detail::ScalarKalman x;
        detail::ScalarKalman y;
        detail::ScalarKalman width;
        detail::ScalarKalman height;

        Track(int track_id, const Detection& detection, double timestamp_sec)
            : id(track_id),
              class_id(detection.class_id),
              score(detection.score),
              last_update_sec(timestamp_sec),
              last_predict_sec(timestamp_sec),
              last_observation(detection) {
            const double box_width = std::max(2.0f, detection.x2 - detection.x1);
            const double box_height = std::max(2.0f, detection.y2 - detection.y1);
            const double center_x = (detection.x1 + detection.x2) * 0.5;
            const double center_y = (detection.y1 + detection.y2) * 0.5;
            const double scale = std::max(box_width, box_height);
            x.initialize(center_x, square(std::max(2.0, scale * 0.05)), square(scale * 0.50));
            y.initialize(center_y, square(std::max(2.0, scale * 0.05)), square(scale * 0.50));
            width.initialize(box_width, square(std::max(2.0, box_width * 0.10)), square(box_width * 0.25));
            height.initialize(box_height, square(std::max(2.0, box_height * 0.10)), square(box_height * 0.25));
        }

        void predict(double timestamp_sec, const CameraMotion& motion) {
            const double dt = std::clamp(timestamp_sec - last_predict_sec, 1.0 / 1000.0, 0.5);
            apply_camera_motion(motion);
            const double scale = std::max({width.position(), height.position(), 4.0});
            const double center_process = square(std::max(0.5, scale * 0.015)) * dt;
            const double center_velocity_process = square(std::max(0.5, scale * 0.010)) * dt;
            const double size_process = square(std::max(0.25, scale * 0.008)) * dt;
            const double size_velocity_process = square(std::max(0.25, scale * 0.004)) * dt;
            x.predict(dt, center_process, center_velocity_process);
            y.predict(dt, center_process, center_velocity_process);
            width.predict(dt, size_process, size_velocity_process);
            height.predict(dt, size_process, size_velocity_process);
            last_predict_sec = timestamp_sec;
            ++age;
        }

        void update(const Detection& detection, double timestamp_sec) {
            const double box_width = std::max(2.0f, detection.x2 - detection.x1);
            const double box_height = std::max(2.0f, detection.y2 - detection.y1);
            const double scale = std::max(box_width, box_height);
            x.update((detection.x1 + detection.x2) * 0.5, center_measurement_variance(scale));
            y.update((detection.y1 + detection.y2) * 0.5, center_measurement_variance(scale));
            width.update(box_width, size_measurement_variance(box_width));
            height.update(box_height, size_measurement_variance(box_height));
            class_id = detection.class_id;
            score = detection.score;
            const double observation_dt = timestamp_sec - last_update_sec;
            if (observation_dt > 1e-6) {
                const std::array<double, 4> previous{
                    last_observation.x1, last_observation.y1,
                    last_observation.x2, last_observation.y2,
                };
                const std::array<double, 4> current{
                    detection.x1, detection.y1, detection.x2, detection.y2,
                };
                for (std::size_t index = 0; index < observation_velocity.size(); ++index) {
                    const double instantaneous = (current[index] - previous[index]) / observation_dt;
                    observation_velocity[index] = has_observation_velocity
                        ? 0.20 * observation_velocity[index] + 0.80 * instantaneous
                        : instantaneous;
                }
                has_observation_velocity = true;
            }
            last_observation = detection;
            last_update_sec = timestamp_sec;
            ++hits;
            ++consecutive_hits;
        }

        void mark_missed(double) { consecutive_hits = 0; }

        double time_since_update(double timestamp_sec) const {
            return std::max(0.0, timestamp_sec - last_update_sec);
        }

        std::array<double, 4> xyxy() const {
            const double elapsed = std::max(0.0, last_predict_sec - last_update_sec);
            std::array<double, 4> box{
                last_observation.x1 + observation_velocity[0] * elapsed,
                last_observation.y1 + observation_velocity[1] * elapsed,
                last_observation.x2 + observation_velocity[2] * elapsed,
                last_observation.y2 + observation_velocity[3] * elapsed,
            };
            if (box[2] < box[0] + 2.0) box[2] = box[0] + 2.0;
            if (box[3] < box[1] + 2.0) box[3] = box[1] + 2.0;
            return box;
        }

        std::array<double, 4> last_observation_xyxy() const {
            return {
                last_observation.x1, last_observation.y1,
                last_observation.x2, last_observation.y2,
            };
        }

        double mahalanobis(const Detection& detection) const {
            const double box_width = std::max(2.0f, detection.x2 - detection.x1);
            const double box_height = std::max(2.0f, detection.y2 - detection.y1);
            const double scale = std::max(box_width, box_height);
            return x.squared_distance((detection.x1 + detection.x2) * 0.5, center_measurement_variance(scale)) +
                   y.squared_distance((detection.y1 + detection.y2) * 0.5, center_measurement_variance(scale)) +
                   width.squared_distance(box_width, size_measurement_variance(box_width)) +
                   height.squared_distance(box_height, size_measurement_variance(box_height));
        }

        TrackOutput output(
            double timestamp_sec,
            bool predicted,
            bool confirmed,
            int image_width,
            int image_height) const {
            std::array<double, 4> box = predicted ? xyxy() : last_observation_xyxy();
            if (image_width > 0) {
                box[0] = std::clamp(box[0], 0.0, static_cast<double>(image_width));
                box[2] = std::clamp(box[2], 0.0, static_cast<double>(image_width));
            }
            if (image_height > 0) {
                box[1] = std::clamp(box[1], 0.0, static_cast<double>(image_height));
                box[3] = std::clamp(box[3], 0.0, static_cast<double>(image_height));
            }
            const double missed_sec = time_since_update(timestamp_sec);
            const float output_score = predicted
                ? static_cast<float>(score * std::exp(-2.0 * missed_sec))
                : score;
            return TrackOutput{
                id, class_id,
                static_cast<float>(box[0]), static_cast<float>(box[1]),
                static_cast<float>(box[2]), static_cast<float>(box[3]),
                output_score, predicted, confirmed, age, hits, missed_sec,
            };
        }

        void apply_camera_motion(const CameraMotion& motion) {
            if (motion.is_identity()) return;
            const auto transform_point = [&](double point_x, double point_y) {
                return std::array<double, 2>{
                    motion.a00 * point_x + motion.a01 * point_y + motion.a02,
                    motion.a10 * point_x + motion.a11 * point_y + motion.a12,
                };
            };
            const std::array<std::array<double, 2>, 4> corners{
                transform_point(last_observation.x1, last_observation.y1),
                transform_point(last_observation.x2, last_observation.y1),
                transform_point(last_observation.x1, last_observation.y2),
                transform_point(last_observation.x2, last_observation.y2),
            };
            double transformed_x1 = corners[0][0];
            double transformed_y1 = corners[0][1];
            double transformed_x2 = corners[0][0];
            double transformed_y2 = corners[0][1];
            for (const auto& corner : corners) {
                transformed_x1 = std::min(transformed_x1, corner[0]);
                transformed_y1 = std::min(transformed_y1, corner[1]);
                transformed_x2 = std::max(transformed_x2, corner[0]);
                transformed_y2 = std::max(transformed_y2, corner[1]);
            }
            last_observation.x1 = static_cast<float>(transformed_x1);
            last_observation.y1 = static_cast<float>(transformed_y1);
            last_observation.x2 = static_cast<float>(transformed_x2);
            last_observation.y2 = static_cast<float>(transformed_y2);

            const double old_observation_vx1 = observation_velocity[0];
            const double old_observation_vy1 = observation_velocity[1];
            const double old_observation_vx2 = observation_velocity[2];
            const double old_observation_vy2 = observation_velocity[3];
            observation_velocity[0] = motion.a00 * old_observation_vx1 + motion.a01 * old_observation_vy1;
            observation_velocity[1] = motion.a10 * old_observation_vx1 + motion.a11 * old_observation_vy1;
            observation_velocity[2] = motion.a00 * old_observation_vx2 + motion.a01 * old_observation_vy2;
            observation_velocity[3] = motion.a10 * old_observation_vx2 + motion.a11 * old_observation_vy2;

            const double old_x = x.position();
            const double old_y = y.position();
            const double old_vx = x.velocity();
            const double old_vy = y.velocity();
            x.set_position_velocity(
                motion.a00 * old_x + motion.a01 * old_y + motion.a02,
                motion.a00 * old_vx + motion.a01 * old_vy);
            y.set_position_velocity(
                motion.a10 * old_x + motion.a11 * old_y + motion.a12,
                motion.a10 * old_vx + motion.a11 * old_vy);
            const double scale_x = std::hypot(motion.a00, motion.a10);
            const double scale_y = std::hypot(motion.a01, motion.a11);
            width.set_position_velocity(width.position() * scale_x, width.velocity() * scale_x);
            height.set_position_velocity(height.position() * scale_y, height.velocity() * scale_y);
        }

        static double square(double value) { return value * value; }
        static double center_measurement_variance(double scale) {
            return square(std::max(2.0, scale * 0.05));
        }
        static double size_measurement_variance(double size) {
            return square(std::max(2.0, size * 0.10));
        }
    };

    double normalize_timestamp(double timestamp_sec) {
        const double fallback_dt = 1.0 / std::max(config_.fallback_fps, 1.0);
        if (!std::isfinite(timestamp_sec)) {
            timestamp_sec = has_timestamp_ ? last_timestamp_sec_ + fallback_dt : 0.0;
        }
        if (has_timestamp_ && timestamp_sec <= last_timestamp_sec_) {
            timestamp_sec = last_timestamp_sec_ + fallback_dt;
        }
        has_timestamp_ = true;
        last_timestamp_sec_ = timestamp_sec;
        return timestamp_sec;
    }

    void associate(
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices,
        const std::vector<Detection>& detections,
        double maximum_cost,
        std::vector<bool>& matched_track,
        std::vector<bool>& matched_detection) {
        std::vector<int> available_detections;
        for (int index : detection_indices) {
            if (!matched_detection[index]) available_detections.push_back(index);
        }
        if (track_indices.empty() || available_detections.empty()) return;

        std::vector<std::vector<double>> costs(
            track_indices.size(), std::vector<double>(available_detections.size(), maximum_cost + 100.0));
        for (std::size_t row = 0; row < track_indices.size(); ++row) {
            for (std::size_t col = 0; col < available_detections.size(); ++col) {
                costs[row][col] = association_cost(
                    tracks_[track_indices[row]], detections[available_detections[col]]);
            }
        }
        const std::vector<int> assignments = detail::hungarian_assignment(costs, maximum_cost);
        for (std::size_t row = 0; row < assignments.size(); ++row) {
            if (assignments[row] < 0) continue;
            const int track_index = track_indices[row];
            const int detection_index = available_detections[assignments[row]];
            tracks_[track_index].update(detections[detection_index], last_timestamp_sec_);
            matched_track[track_index] = true;
            matched_detection[detection_index] = true;
        }
    }

    static double association_cost(const Track& track, const Detection& detection) {
        if (track.class_id != detection.class_id) return 1000.0;
        const std::array<double, 4> box = track.xyxy();
        const std::array<double, 4> observed_box = track.last_observation_xyxy();
        const double iou = std::max(
            detail::box_iou(box, detection), detail::box_iou(observed_box, detection));
        const double track_width = std::max(2.0, box[2] - box[0]);
        const double track_height = std::max(2.0, box[3] - box[1]);
        const double detection_width = std::max(2.0f, detection.x2 - detection.x1);
        const double detection_height = std::max(2.0f, detection.y2 - detection.y1);
        const double center_dx = (box[0] + box[2] - detection.x1 - detection.x2) * 0.5;
        const double center_dy = (box[1] + box[3] - detection.y1 - detection.y2) * 0.5;
        const double observed_center_dx =
            (observed_box[0] + observed_box[2] - detection.x1 - detection.x2) * 0.5;
        const double observed_center_dy =
            (observed_box[1] + observed_box[3] - detection.y1 - detection.y2) * 0.5;
        const double normalization = std::max(
            4.0, 0.5 * (std::hypot(track_width, track_height) +
                        std::hypot(detection_width, detection_height)));
        const double normalized_center = std::min(
            std::hypot(center_dx, center_dy),
            std::hypot(observed_center_dx, observed_center_dy)) / normalization;
        const double mahalanobis = track.mahalanobis(detection);
        if (iou < 0.01 && normalized_center > 3.0) return 1000.0;

        const double center_similarity = std::exp(-2.0 * normalized_center * normalized_center);
        const double mahalanobis_similarity = std::exp(-0.5 * mahalanobis / 9.4877);
        const double confidence = std::clamp(static_cast<double>(detection.score), 0.0, 1.0);
        const double similarity =
            0.50 * iou + 0.20 * center_similarity + 0.20 * mahalanobis_similarity + 0.10 * confidence;
        return 1.0 - similarity;
    }

    Config config_;
    std::vector<Track> tracks_;
    int next_track_id_ = 1;
    bool has_timestamp_ = false;
    double last_timestamp_sec_ = 0.0;
};

}  // namespace rk_tracker
