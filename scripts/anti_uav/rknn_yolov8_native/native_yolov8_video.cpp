#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <pthread.h>

#include "rknn_api.h"
#include "detector_based_tracker.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
};

struct LetterboxInfo {
    float ratio = 1.0f;
    float dw = 0.0f;
    float dh = 0.0f;
};

struct StageTimes {
    std::vector<double> preprocess_ms;
    std::vector<double> inference_ms;
    std::vector<double> postprocess_ms;
    std::vector<double> tracker_ms;
    std::vector<double> total_ms;
};

struct FrameResult {
    int frame_index = 0;
    bool measured = false;
    double timestamp_sec = 0.0;
    int source_width = 0;
    int source_height = 0;
    std::vector<Detection> detections;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
};

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double ratio) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double index = ratio * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(index);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = index - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Unable to open model: " + path);
    }
    const auto size = input.tellg();
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (!input.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Unable to read model: " + path);
    }
    return data;
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value) {
        if (ch == '\\' || ch == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

rknn_core_mask parse_core_mask(const std::string& value) {
    if (value == "0") return RKNN_NPU_CORE_0;
    if (value == "1") return RKNN_NPU_CORE_1;
    if (value == "2") return RKNN_NPU_CORE_2;
    if (value == "0_1") return RKNN_NPU_CORE_0_1;
    if (value == "0_1_2" || value == "all") return RKNN_NPU_CORE_0_1_2;
    return RKNN_NPU_CORE_AUTO;
}

float tensor_value_i8(
    const rknn_tensor_attr& attr,
    const int8_t* data,
    int channel,
    int y,
    int x,
    int logical_grid_width) {
    size_t index = 0;
    if (attr.fmt == RKNN_TENSOR_NC1HWC2 && attr.n_dims == 5) {
        const int height = attr.dims[2];
        const int width = attr.dims[3];
        const int c2 = attr.dims[4];
        const size_t spatial_index = static_cast<size_t>(y) * logical_grid_width + x;
        index = (static_cast<size_t>(channel / c2) * height * width + spatial_index) * c2 + channel % c2;
    } else if (attr.fmt == RKNN_TENSOR_NCHW && attr.n_dims >= 4) {
        const int height = attr.dims[2];
        const int width = attr.dims[3];
        index = (static_cast<size_t>(channel) * height + y) * width + x;
    } else if (attr.fmt == RKNN_TENSOR_NHWC && attr.n_dims >= 4) {
        const int width = attr.dims[2];
        const int channels = attr.dims[3];
        index = (static_cast<size_t>(y) * width + x) * channels + channel;
    } else {
        throw std::runtime_error("Unsupported native output tensor layout");
    }
    if (index >= attr.size_with_stride) {
        throw std::runtime_error(
            "Native output index exceeds buffer: index=" + std::to_string(index) +
            " size=" + std::to_string(attr.size_with_stride));
    }
    return (static_cast<int32_t>(data[index]) - attr.zp) * attr.scale;
}

float dfl_expectation(
    const rknn_tensor_attr& attr,
    const int8_t* data,
    int side,
    int y,
    int x,
    int logical_grid_width) {
    float logits[16];
    float max_logit = -1e30f;
    for (int bin = 0; bin < 16; ++bin) {
        logits[bin] = tensor_value_i8(attr, data, side * 16 + bin, y, x, logical_grid_width);
        max_logit = std::max(max_logit, logits[bin]);
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (int bin = 0; bin < 16; ++bin) {
        const float probability = std::exp(logits[bin] - max_logit);
        denominator += probability;
        numerator += probability * static_cast<float>(bin);
    }
    return numerator / std::max(denominator, 1e-12f);
}

float overlap_iou(const Detection& lhs, const Detection& rhs) {
    const float x1 = std::max(lhs.x1, rhs.x1);
    const float y1 = std::max(lhs.y1, rhs.y1);
    const float x2 = std::min(lhs.x2, rhs.x2);
    const float y2 = std::min(lhs.y2, rhs.y2);
    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float lhs_area = std::max(0.0f, lhs.x2 - lhs.x1) * std::max(0.0f, lhs.y2 - lhs.y1);
    const float rhs_area = std::max(0.0f, rhs.x2 - rhs.x1) * std::max(0.0f, rhs.y2 - rhs.y1);
    return intersection / std::max(lhs_area + rhs_area - intersection, 1e-12f);
}

std::vector<Detection> nms(std::vector<Detection> candidates, float iou_threshold, int max_detections) {
    std::sort(candidates.begin(), candidates.end(), [](const Detection& lhs, const Detection& rhs) {
        return lhs.score > rhs.score;
    });
    std::vector<Detection> kept;
    kept.reserve(std::min<int>(max_detections, candidates.size()));
    for (const Detection& candidate : candidates) {
        bool suppressed = false;
        for (const Detection& selected : kept) {
            if (candidate.class_id == selected.class_id && overlap_iou(candidate, selected) > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
            if (static_cast<int>(kept.size()) >= max_detections) {
                break;
            }
        }
    }
    return kept;
}

class NativeYoloV8 {
public:
    NativeYoloV8(const std::string& model_path, const std::string& core_mask, bool defer_io = false)
        : model_data_(read_file(model_path)) {
        int ret = rknn_init(&context_, model_data_.data(), static_cast<uint32_t>(model_data_.size()), 0, nullptr);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("rknn_init failed: " + std::to_string(ret));
        }
        if (!defer_io) initialize_io(core_mask);
    }

    NativeYoloV8(NativeYoloV8& source, const std::string& core_mask, bool defer_io = false) {
        const int ret = rknn_dup_context(&source.context_, &context_);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("rknn_dup_context failed: " + std::to_string(ret));
        }
        if (!defer_io) initialize_io(core_mask);
    }

    ~NativeYoloV8() {
        for (rknn_tensor_mem* memory : output_mems_) {
            if (memory != nullptr) rknn_destroy_mem(context_, memory);
        }
        if (input_mem_ != nullptr) rknn_destroy_mem(context_, input_mem_);
        if (context_ != 0) rknn_destroy(context_);
    }

    NativeYoloV8(const NativeYoloV8&) = delete;
    NativeYoloV8& operator=(const NativeYoloV8&) = delete;

    void initialize_deferred_io(const std::string& core_mask) {
        if (io_initialized_) throw std::runtime_error("RKNN I/O already initialized");
        initialize_io(core_mask);
    }

private:
    void initialize_io(const std::string& core_mask) {
        int ret = rknn_set_core_mask(context_, parse_core_mask(core_mask));
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("rknn_set_core_mask failed: " + std::to_string(ret));
        }

        ret = rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
        if (ret != RKNN_SUCC || io_num_.n_input != 1 ||
            (io_num_.n_output != 9 && io_num_.n_output != 12)) {
            throw std::runtime_error("Expected one input and 9 (P3-P5) or 12 (P2-P5) RK-optimized YOLO outputs");
        }
        branch_count_ = static_cast<int>(io_num_.n_output / 3);

        input_attr_.index = 0;
        ret = rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));
        if (ret != RKNN_SUCC) throw std::runtime_error("RKNN_QUERY_INPUT_ATTR failed");
        input_native_attr_.index = 0;
        ret = rknn_query(context_, RKNN_QUERY_NATIVE_INPUT_ATTR, &input_native_attr_, sizeof(input_native_attr_));
        if (ret != RKNN_SUCC) throw std::runtime_error("RKNN_QUERY_NATIVE_INPUT_ATTR failed");

        if (input_attr_.fmt == RKNN_TENSOR_NHWC) {
            input_height_ = input_attr_.dims[1];
            input_width_ = input_attr_.dims[2];
        } else {
            input_height_ = input_attr_.dims[2];
            input_width_ = input_attr_.dims[3];
        }
        input_native_attr_.type = RKNN_TENSOR_UINT8;
        input_mem_ = rknn_create_mem(context_, input_native_attr_.size_with_stride);
        if (input_mem_ == nullptr) throw std::runtime_error("rknn_create_mem input failed");
        ret = rknn_set_io_mem(context_, input_mem_, &input_native_attr_);
        if (ret != RKNN_SUCC) throw std::runtime_error("rknn_set_io_mem input failed");

        output_attrs_.resize(io_num_.n_output);
        output_native_attrs_.resize(io_num_.n_output);
        output_mems_.resize(io_num_.n_output, nullptr);
        for (uint32_t index = 0; index < io_num_.n_output; ++index) {
            output_attrs_[index].index = index;
            output_native_attrs_[index].index = index;
            if (rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[index], sizeof(rknn_tensor_attr)) != RKNN_SUCC ||
                rknn_query(context_, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &output_native_attrs_[index], sizeof(rknn_tensor_attr)) != RKNN_SUCC) {
                throw std::runtime_error("RKNN output attribute query failed");
            }
            if (output_native_attrs_[index].type != RKNN_TENSOR_INT8) {
                throw std::runtime_error("Native output is not INT8");
            }
            output_mems_[index] = rknn_create_mem(context_, output_native_attrs_[index].size_with_stride);
            if (output_mems_[index] == nullptr ||
                rknn_set_io_mem(context_, output_mems_[index], &output_native_attrs_[index]) != RKNN_SUCC) {
                throw std::runtime_error("RKNN output zero-copy setup failed");
            }
        }
        io_initialized_ = true;
    }

public:

    LetterboxInfo preprocess(const cv::Mat& frame_bgr) {
        const float ratio = std::min(
            static_cast<float>(input_height_) / frame_bgr.rows,
            static_cast<float>(input_width_) / frame_bgr.cols);
        const int resized_width = std::max(1, static_cast<int>(std::round(frame_bgr.cols * ratio)));
        const int resized_height = std::max(1, static_cast<int>(std::round(frame_bgr.rows * ratio)));
        const float dw = static_cast<float>(input_width_ - resized_width) * 0.5f;
        const float dh = static_cast<float>(input_height_ - resized_height) * 0.5f;
        const int left = static_cast<int>(std::round(dw - 0.1f));
        const int top = static_cast<int>(std::round(dh - 0.1f));

        const int row_stride = input_native_attr_.w_stride > 0 ? input_native_attr_.w_stride * 3 : input_width_ * 3;
        cv::Mat input_rgb(input_height_, input_width_, CV_8UC3, input_mem_->virt_addr, row_stride);
        input_rgb.setTo(cv::Scalar(0, 0, 0));
        cv::resize(frame_bgr, resized_bgr_, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
        cv::Mat destination = input_rgb(cv::Rect(left, top, resized_width, resized_height));
        cv::cvtColor(resized_bgr_, destination, cv::COLOR_BGR2RGB);
        return LetterboxInfo{ratio, dw, dh};
    }

    void run() {
        const int ret = rknn_run(context_, nullptr);
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("rknn_run failed: " + std::to_string(ret));
        }
    }

    std::vector<Detection> decode(
        const LetterboxInfo& letterbox,
        int source_width,
        int source_height,
        float confidence_threshold,
        float nms_threshold,
        int max_detections) const {
        std::vector<Detection> candidates;
        candidates.reserve(128);
        for (int branch = 0; branch < branch_count_; ++branch) {
            const int box_index = branch * 3;
            const int class_index = box_index + 1;
            const int sum_index = box_index + 2;
            const rknn_tensor_attr& box_attr = output_native_attrs_[box_index];
            const rknn_tensor_attr& class_attr = output_native_attrs_[class_index];
            const rknn_tensor_attr& sum_attr = output_native_attrs_[sum_index];
            const int8_t* box_data = static_cast<const int8_t*>(output_mems_[box_index]->virt_addr);
            const int8_t* class_data = static_cast<const int8_t*>(output_mems_[class_index]->virt_addr);
            const int8_t* sum_data = static_cast<const int8_t*>(output_mems_[sum_index]->virt_addr);
            const int grid_height = output_attrs_[box_index].dims[2];
            const int grid_width = output_attrs_[box_index].dims[3];
            const float stride = static_cast<float>(input_height_) / grid_height;

            for (int y = 0; y < grid_height; ++y) {
                for (int x = 0; x < grid_width; ++x) {
                    const float class_score = tensor_value_i8(class_attr, class_data, 0, y, x, grid_width);
                    if (class_score < confidence_threshold ||
                        tensor_value_i8(sum_attr, sum_data, 0, y, x, grid_width) < confidence_threshold) {
                        continue;
                    }
                    const float left = dfl_expectation(box_attr, box_data, 0, y, x, grid_width);
                    const float top = dfl_expectation(box_attr, box_data, 1, y, x, grid_width);
                    const float right = dfl_expectation(box_attr, box_data, 2, y, x, grid_width);
                    const float bottom = dfl_expectation(box_attr, box_data, 3, y, x, grid_width);
                    Detection detection{};
                    detection.x1 = ((x + 0.5f - left) * stride - letterbox.dw) / letterbox.ratio;
                    detection.y1 = ((y + 0.5f - top) * stride - letterbox.dh) / letterbox.ratio;
                    detection.x2 = ((x + 0.5f + right) * stride - letterbox.dw) / letterbox.ratio;
                    detection.y2 = ((y + 0.5f + bottom) * stride - letterbox.dh) / letterbox.ratio;
                    detection.x1 = std::clamp(detection.x1, 0.0f, static_cast<float>(source_width));
                    detection.y1 = std::clamp(detection.y1, 0.0f, static_cast<float>(source_height));
                    detection.x2 = std::clamp(detection.x2, 0.0f, static_cast<float>(source_width));
                    detection.y2 = std::clamp(detection.y2, 0.0f, static_cast<float>(source_height));
                    detection.score = class_score;
                    detection.class_id = 0;
                    candidates.push_back(detection);
                }
            }
        }
        return nms(std::move(candidates), nms_threshold, max_detections);
    }

    int input_height() const { return input_height_; }
    int input_width() const { return input_width_; }

private:
    std::vector<uint8_t> model_data_;
    rknn_context context_ = 0;
    rknn_input_output_num io_num_{};
    rknn_tensor_attr input_attr_{};
    rknn_tensor_attr input_native_attr_{};
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<rknn_tensor_attr> output_native_attrs_;
    rknn_tensor_mem* input_mem_ = nullptr;
    std::vector<rknn_tensor_mem*> output_mems_;
    int input_height_ = 0;
    int input_width_ = 0;
    int branch_count_ = 0;
    cv::Mat resized_bgr_;
    bool io_initialized_ = false;
};

void pin_current_thread(int cpu_index) {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu_index, &cpu_set);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
}

class DetectorWorker {
public:
    using ResultCallback = std::function<void(FrameResult&&)>;

    DetectorWorker(
        int worker_index,
        int cpu_index,
        std::unique_ptr<NativeYoloV8> detector,
        int queue_size,
        float confidence,
        float nms_iou,
        int max_detections,
        ResultCallback callback)
        : worker_index_(worker_index),
          cpu_index_(cpu_index),
          detector_(std::move(detector)),
          queue_size_(queue_size),
          confidence_(confidence),
          nms_iou_(nms_iou),
          max_detections_(max_detections),
          callback_(std::move(callback)),
          thread_(&DetectorWorker::run_loop, this) {}

    ~DetectorWorker() {
        close();
    }

    int input_height() const { return detector_->input_height(); }
    int input_width() const { return detector_->input_width(); }

    void enqueue(int frame_index, bool measured, double timestamp_sec, cv::Mat&& frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return closed_ || static_cast<int>(queue_.size()) < queue_size_; });
        if (closed_) throw std::runtime_error("enqueue on closed detector worker");
        queue_.push_back(Task{frame_index, measured, timestamp_sec, std::move(frame)});
        not_empty_.notify_one();
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                if (thread_.joinable()) thread_.join();
                return;
            }
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    struct Task {
        int frame_index;
        bool measured;
        double timestamp_sec;
        cv::Mat frame;
    };

    void run_loop() {
        pin_current_thread(cpu_index_);
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                not_empty_.wait(lock, [this] { return closed_ || !queue_.empty(); });
                if (queue_.empty()) {
                    if (closed_) return;
                    continue;
                }
                task = std::move(queue_.front());
                queue_.pop_front();
                not_full_.notify_one();
            }

            FrameResult result;
            result.frame_index = task.frame_index;
            result.measured = task.measured;
            result.timestamp_sec = task.timestamp_sec;
            result.source_width = task.frame.cols;
            result.source_height = task.frame.rows;
            const auto total_start = Clock::now();
            const auto preprocess_start = Clock::now();
            const LetterboxInfo letterbox = detector_->preprocess(task.frame);
            result.preprocess_ms = elapsed_ms(preprocess_start);
            const auto inference_start = Clock::now();
            detector_->run();
            result.inference_ms = elapsed_ms(inference_start);
            const auto postprocess_start = Clock::now();
            result.detections = detector_->decode(
                letterbox, task.frame.cols, task.frame.rows, confidence_, nms_iou_, max_detections_);
            result.postprocess_ms = elapsed_ms(postprocess_start);
            result.total_ms = elapsed_ms(total_start);
            callback_(std::move(result));
        }
    }

    int worker_index_;
    int cpu_index_;
    std::unique_ptr<NativeYoloV8> detector_;
    int queue_size_;
    float confidence_;
    float nms_iou_;
    int max_detections_;
    ResultCallback callback_;
    std::deque<Task> queue_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    bool closed_ = false;
    std::thread thread_;
};

struct Options {
    std::string model;
    std::string video;
    std::string output_json;
    std::string predictions_csv;
    std::string tracks_csv;
    std::string core_mask = "0_1_2";
    std::string tracker = "none";
    int max_frames = 0;
    int warmup_frames = 50;
    int max_detections = 128;
    int workers = 1;
    int queue_size = 3;
    int worker_cpu_base = 4;
    float confidence = 0.45f;
    float nms_iou = 0.45f;
    float track_high_threshold = 0.45f;
    float track_low_threshold = 0.10f;
    float new_track_threshold = 0.50f;
    float track_first_match_cost = 0.92f;
    float track_second_match_cost = 0.92f;
    double track_buffer_sec = 1.0;
    double track_prediction_sec = 0.0;
    double source_fps = 0.0;
    int track_min_hits = 2;
};

Options parse_options(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: native_yolov8_video MODEL VIDEO [--output-json PATH] [--predictions-csv PATH] "
            "[--core-mask 0_1_2] [--workers N] [--queue-size N] [--worker-cpu-base N] "
            "[--tracker none|rk_botsort] [--tracks-csv PATH] [--source-fps F] "
            "[--max-frames N] [--warmup-frames N] [--conf F] [--nms-iou F]");
    }
    Options options;
    options.model = argv[1];
    options.video = argv[2];
    for (int index = 3; index < argc; ++index) {
        const std::string key = argv[index];
        if (index + 1 >= argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[++index];
        if (key == "--output-json") options.output_json = value;
        else if (key == "--predictions-csv") options.predictions_csv = value;
        else if (key == "--tracks-csv") options.tracks_csv = value;
        else if (key == "--core-mask") options.core_mask = value;
        else if (key == "--tracker") options.tracker = value;
        else if (key == "--max-frames") options.max_frames = std::stoi(value);
        else if (key == "--warmup-frames") options.warmup_frames = std::stoi(value);
        else if (key == "--max-det") options.max_detections = std::stoi(value);
        else if (key == "--workers") options.workers = std::stoi(value);
        else if (key == "--queue-size") options.queue_size = std::stoi(value);
        else if (key == "--worker-cpu-base") options.worker_cpu_base = std::stoi(value);
        else if (key == "--conf") options.confidence = std::stof(value);
        else if (key == "--nms-iou") options.nms_iou = std::stof(value);
        else if (key == "--track-high-thresh") options.track_high_threshold = std::stof(value);
        else if (key == "--track-low-thresh") options.track_low_threshold = std::stof(value);
        else if (key == "--new-track-thresh") options.new_track_threshold = std::stof(value);
        else if (key == "--track-first-match-cost") options.track_first_match_cost = std::stof(value);
        else if (key == "--track-second-match-cost") options.track_second_match_cost = std::stof(value);
        else if (key == "--track-buffer-sec") options.track_buffer_sec = std::stod(value);
        else if (key == "--track-prediction-sec") options.track_prediction_sec = std::stod(value);
        else if (key == "--track-min-hits") options.track_min_hits = std::stoi(value);
        else if (key == "--source-fps") options.source_fps = std::stod(value);
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (options.workers < 1 || options.workers > 3) throw std::runtime_error("--workers must be between 1 and 3");
    if (options.queue_size < 1) throw std::runtime_error("--queue-size must be positive");
    if (options.worker_cpu_base < 0 || options.worker_cpu_base + options.workers > 8) {
        throw std::runtime_error("worker CPU range must fit within CPU 0-7");
    }
    if (options.tracker != "none" && options.tracker != "rk_botsort") {
        throw std::runtime_error("--tracker must be none or rk_botsort");
    }
    if (!options.tracks_csv.empty() && options.tracker == "none") options.tracker = "rk_botsort";
    if (!(options.track_low_threshold >= 0.0f &&
          options.track_low_threshold < options.track_high_threshold &&
          options.track_high_threshold <= 1.0f)) {
        throw std::runtime_error("tracker thresholds must satisfy 0 <= low < high <= 1");
    }
    if (options.track_buffer_sec <= 0.0 || options.track_prediction_sec < 0.0 || options.track_min_hits < 1) {
        throw std::runtime_error("invalid tracker lifetime configuration");
    }
    return options;
}

struct TrackingStats {
    int frames_with_tracks = 0;
    int output_count = 0;
    int predicted_count = 0;
    std::set<int> unique_track_ids;
};

double resolve_source_fps(const Options& options, cv::VideoCapture& capture) {
    if (options.source_fps > 0.0) return options.source_fps;
    const double detected_fps = capture.get(cv::CAP_PROP_FPS);
    return std::isfinite(detected_fps) && detected_fps > 0.0 ? detected_fps : 30.0;
}

double frame_timestamp_sec(
    const Options& options,
    cv::VideoCapture& capture,
    int frame_index,
    double source_fps) {
    if (options.source_fps > 0.0) return static_cast<double>(frame_index) / source_fps;
    const double position_ms = capture.get(cv::CAP_PROP_POS_MSEC);
    if (std::isfinite(position_ms) && (frame_index == 0 || position_ms > 0.0)) return position_ms / 1000.0;
    return static_cast<double>(frame_index) / source_fps;
}

float detector_confidence(const Options& options) {
    return options.tracker == "none"
        ? options.confidence
        : std::min(options.confidence, options.track_low_threshold);
}

class ResultConsumer {
public:
    ResultConsumer(const Options& options, double source_fps) {
        if (!options.predictions_csv.empty()) {
            predictions_.open(options.predictions_csv, std::ios::out | std::ios::trunc);
            if (!predictions_) throw std::runtime_error("Unable to open predictions CSV");
            predictions_ << "frame,x,y,width,height,confidence,class_id\n";
            predictions_ << std::fixed << std::setprecision(6);
        }
        if (!options.tracks_csv.empty()) {
            tracks_output_.open(options.tracks_csv, std::ios::out | std::ios::trunc);
            if (!tracks_output_) throw std::runtime_error("Unable to open tracks CSV");
            tracks_output_ << "frame,timestamp_sec,track_id,x,y,width,height,confidence,class_id,"
                              "predicted,confirmed,age,hits,time_since_update_sec\n";
            tracks_output_ << std::fixed << std::setprecision(6);
        }
        if (options.tracker == "rk_botsort") {
            rk_tracker::Config config;
            config.high_threshold = options.track_high_threshold;
            config.low_threshold = options.track_low_threshold;
            config.new_track_threshold = options.new_track_threshold;
            config.first_match_cost = options.track_first_match_cost;
            config.second_match_cost = options.track_second_match_cost;
            config.track_buffer_sec = options.track_buffer_sec;
            config.prediction_grace_sec = options.track_prediction_sec;
            config.fallback_fps = source_fps;
            config.min_hits = options.track_min_hits;
            tracker_ = std::make_unique<rk_tracker::DetectorBasedTracker>(config);
        }
    }

    void consume(const FrameResult& result) {
        if (!result.detections.empty()) {
            ++detection_frames_;
            detection_count_ += static_cast<int>(result.detections.size());
            if (predictions_) {
                const Detection& best = result.detections.front();
                predictions_ << result.frame_index << ',' << best.x1 << ',' << best.y1 << ','
                             << best.x2 - best.x1 << ',' << best.y2 - best.y1 << ','
                             << best.score << ',' << best.class_id << '\n';
            }
        }

        double tracker_ms = 0.0;
        if (tracker_) {
            std::vector<rk_tracker::Detection> tracker_detections;
            tracker_detections.reserve(result.detections.size());
            for (const Detection& detection : result.detections) {
                tracker_detections.push_back(rk_tracker::Detection{
                    detection.x1, detection.y1, detection.x2, detection.y2,
                    detection.score, detection.class_id,
                });
            }
            const auto tracker_start = Clock::now();
            const std::vector<rk_tracker::TrackOutput> tracks = tracker_->update(
                tracker_detections, result.timestamp_sec, result.source_width, result.source_height);
            tracker_ms = elapsed_ms(tracker_start);
            if (!tracks.empty()) ++tracking_stats_.frames_with_tracks;
            tracking_stats_.output_count += static_cast<int>(tracks.size());
            for (const rk_tracker::TrackOutput& track : tracks) {
                tracking_stats_.unique_track_ids.insert(track.track_id);
                if (track.predicted) ++tracking_stats_.predicted_count;
                if (tracks_output_) {
                    tracks_output_ << result.frame_index << ',' << result.timestamp_sec << ',' << track.track_id << ','
                                   << track.x1 << ',' << track.y1 << ',' << track.x2 - track.x1 << ','
                                   << track.y2 - track.y1 << ',' << track.score << ',' << track.class_id << ','
                                   << (track.predicted ? 1 : 0) << ',' << (track.confirmed ? 1 : 0) << ','
                                   << track.age << ',' << track.hits << ',' << track.time_since_update_sec << '\n';
                }
            }
        }

        if (result.measured) {
            times_.preprocess_ms.push_back(result.preprocess_ms);
            times_.inference_ms.push_back(result.inference_ms);
            times_.postprocess_ms.push_back(result.postprocess_ms);
            times_.tracker_ms.push_back(tracker_ms);
            times_.total_ms.push_back(result.total_ms + tracker_ms);
        }
    }

    const StageTimes& times() const { return times_; }
    const TrackingStats& tracking_stats() const { return tracking_stats_; }
    int detection_frames() const { return detection_frames_; }
    int detection_count() const { return detection_count_; }

private:
    std::unique_ptr<rk_tracker::DetectorBasedTracker> tracker_;
    std::ofstream predictions_;
    std::ofstream tracks_output_;
    StageTimes times_;
    TrackingStats tracking_stats_;
    int detection_frames_ = 0;
    int detection_count_ = 0;
};

void write_summary(
    std::ostream& output,
    const Options& options,
    int input_height,
    int input_width,
    int total_frames,
    int measured_frames,
    int detection_frames,
    int detection_count,
    const StageTimes& times,
    const TrackingStats& tracking_stats,
    int workers = 1,
    double throughput_wall_ms = 0.0) {
    const double total_mean = mean(times.total_ms);
    const double throughput_fps = throughput_wall_ms > 0.0
        ? static_cast<double>(measured_frames) * 1000.0 / throughput_wall_ms
        : 1000.0 / total_mean;
    output << std::fixed << std::setprecision(6);
    output << "{\n";
    output << "  \"model\": \"" << json_escape(options.model) << "\",\n";
    output << "  \"source\": \"" << json_escape(options.video) << "\",\n";
    output << "  \"input_size\": [" << input_height << ", " << input_width << "],\n";
    output << "  \"runtime\": \"native_capi_zero_copy_nc1hwc2\",\n";
    output << "  \"tracker\": \"" << options.tracker << "\",\n";
    output << "  \"core_mask\": \"" << options.core_mask << "\",\n";
    output << "  \"workers\": " << workers << ",\n";
    output << "  \"worker_cpu_base\": " << options.worker_cpu_base << ",\n";
    output << "  \"total_frames\": " << total_frames << ",\n";
    output << "  \"measured_frames\": " << measured_frames << ",\n";
    output << "  \"warmup_frames\": " << options.warmup_frames << ",\n";
    output << "  \"mean_ms\": " << total_mean << ",\n";
    output << "  \"p50_ms\": " << percentile(times.total_ms, 0.50) << ",\n";
    output << "  \"p95_ms\": " << percentile(times.total_ms, 0.95) << ",\n";
    output << "  \"max_ms\": " << *std::max_element(times.total_ms.begin(), times.total_ms.end()) << ",\n";
    output << "  \"fps_mean\": " << throughput_fps << ",\n";
    output << "  \"single_frame_fps\": " << 1000.0 / total_mean << ",\n";
    output << "  \"throughput_wall_ms\": " << throughput_wall_ms << ",\n";
    output << "  \"throughput_includes_video_read\": " << (workers > 1 ? "true" : "false") << ",\n";
    output << "  \"detection_frames\": " << detection_frames << ",\n";
    output << "  \"detection_count\": " << detection_count << ",\n";
    output << "  \"tracking_frames\": " << tracking_stats.frames_with_tracks << ",\n";
    output << "  \"tracking_outputs\": " << tracking_stats.output_count << ",\n";
    output << "  \"tracking_predicted_outputs\": " << tracking_stats.predicted_count << ",\n";
    output << "  \"unique_track_ids\": " << tracking_stats.unique_track_ids.size() << ",\n";
    output << "  \"detail_mean_ms\": {\n";
    output << "    \"preprocess_letterbox\": " << mean(times.preprocess_ms) << ",\n";
    output << "    \"inference\": " << mean(times.inference_ms) << ",\n";
    output << "    \"native_decode_nms\": " << mean(times.postprocess_ms) << ",\n";
    output << "    \"tracker_association\": " << mean(times.tracker_ms) << "\n";
    output << "  }\n";
    output << "}\n";
}

int run_parallel(const Options& options) {
    cv::setNumThreads(1);
    cv::VideoCapture capture(options.video);
    if (!capture.isOpened()) throw std::runtime_error("Unable to open video: " + options.video);
    const double source_fps = resolve_source_fps(options, capture);
    ResultConsumer consumer(options, source_fps);

    std::mutex result_mutex;
    std::condition_variable result_ready;
    std::map<int, FrameResult> pending_results;
    int completed = 0;
    int consumed = 0;
    bool producers_done = false;
    std::exception_ptr consumer_error;
    auto callback = [&](FrameResult&& result) {
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            pending_results.emplace(result.frame_index, std::move(result));
            ++completed;
        }
        result_ready.notify_all();
    };

    std::thread consumer_thread([&] {
        try {
            int next_frame = 0;
            while (true) {
                FrameResult result;
                {
                    std::unique_lock<std::mutex> lock(result_mutex);
                    result_ready.wait(lock, [&] {
                        return producers_done || pending_results.find(next_frame) != pending_results.end();
                    });
                    auto iterator = pending_results.find(next_frame);
                    if (iterator == pending_results.end()) {
                        if (producers_done && pending_results.empty()) break;
                        if (producers_done) throw std::runtime_error("Missing detector result before end of stream");
                        continue;
                    }
                    result = std::move(iterator->second);
                    pending_results.erase(iterator);
                }
                consumer.consume(result);
                {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    ++consumed;
                }
                result_ready.notify_all();
                ++next_frame;
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(result_mutex);
                consumer_error = std::current_exception();
            }
            result_ready.notify_all();
        }
    });

    const char* masks[] = {"0", "1", "2"};
    std::vector<std::unique_ptr<NativeYoloV8>> detectors;
    detectors.push_back(std::make_unique<NativeYoloV8>(options.model, masks[0], true));
    for (int index = 1; index < options.workers; ++index) {
        detectors.push_back(std::make_unique<NativeYoloV8>(*detectors.front(), masks[index], true));
    }
    for (int index = 0; index < options.workers; ++index) {
        detectors[index]->initialize_deferred_io(masks[index]);
    }
    std::vector<std::unique_ptr<DetectorWorker>> workers;
    for (int index = 0; index < options.workers; ++index) {
        workers.push_back(std::make_unique<DetectorWorker>(
            index, options.worker_cpu_base + index, std::move(detectors[index]), options.queue_size,
            detector_confidence(options),
            options.nms_iou, options.max_detections, callback));
    }
    const int input_height = workers.front()->input_height();
    const int input_width = workers.front()->input_width();

    cv::Mat frame;
    int frame_index = 0;
    while (frame_index < options.warmup_frames && capture.read(frame)) {
        const double timestamp_sec = frame_timestamp_sec(options, capture, frame_index, source_fps);
        workers[frame_index % options.workers]->enqueue(frame_index, false, timestamp_sec, std::move(frame));
        ++frame_index;
    }
    {
        std::unique_lock<std::mutex> lock(result_mutex);
        result_ready.wait(lock, [&] { return consumer_error || consumed >= frame_index; });
        if (consumer_error) std::rethrow_exception(consumer_error);
    }

    const int warmup_completed = frame_index;
    const auto throughput_start = Clock::now();
    while ((options.max_frames <= 0 || frame_index < options.max_frames) && capture.read(frame)) {
        const double timestamp_sec = frame_timestamp_sec(options, capture, frame_index, source_fps);
        workers[frame_index % options.workers]->enqueue(frame_index, true, timestamp_sec, std::move(frame));
        ++frame_index;
    }
    for (auto& worker : workers) worker->close();
    {
        std::lock_guard<std::mutex> lock(result_mutex);
        producers_done = true;
    }
    result_ready.notify_all();
    consumer_thread.join();
    const double throughput_wall_ms = elapsed_ms(throughput_start);
    if (consumer_error) std::rethrow_exception(consumer_error);
    if (completed != frame_index || consumed != frame_index) {
        throw std::runtime_error("Not all detector results were consumed");
    }

    const StageTimes& times = consumer.times();
    if (times.total_ms.empty()) throw std::runtime_error("No measured frames");

    write_summary(
        std::cout, options, input_height, input_width, frame_index, frame_index - warmup_completed,
        consumer.detection_frames(), consumer.detection_count(), times, consumer.tracking_stats(),
        options.workers, throughput_wall_ms);
    if (!options.output_json.empty()) {
        std::ofstream summary(options.output_json, std::ios::out | std::ios::trunc);
        if (!summary) throw std::runtime_error("Unable to open output JSON");
        write_summary(
            summary, options, input_height, input_width, frame_index, frame_index - warmup_completed,
            consumer.detection_frames(), consumer.detection_count(), times, consumer.tracking_stats(),
            options.workers, throughput_wall_ms);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.workers > 1) {
            return run_parallel(options);
        }
        NativeYoloV8 detector(options.model, options.core_mask);
        cv::VideoCapture capture(options.video);
        if (!capture.isOpened()) throw std::runtime_error("Unable to open video: " + options.video);
        const double source_fps = resolve_source_fps(options, capture);
        ResultConsumer consumer(options, source_fps);
        int frame_index = 0;
        cv::Mat frame;
        while (capture.read(frame)) {
            if (options.max_frames > 0 && frame_index >= options.max_frames) break;
            FrameResult result;
            result.frame_index = frame_index;
            result.measured = frame_index >= options.warmup_frames;
            result.timestamp_sec = frame_timestamp_sec(options, capture, frame_index, source_fps);
            result.source_width = frame.cols;
            result.source_height = frame.rows;
            const auto total_start = Clock::now();
            const auto preprocess_start = Clock::now();
            const LetterboxInfo letterbox = detector.preprocess(frame);
            result.preprocess_ms = elapsed_ms(preprocess_start);
            const auto inference_start = Clock::now();
            detector.run();
            result.inference_ms = elapsed_ms(inference_start);
            const auto postprocess_start = Clock::now();
            result.detections = detector.decode(
                letterbox, frame.cols, frame.rows, detector_confidence(options),
                options.nms_iou, options.max_detections);
            result.postprocess_ms = elapsed_ms(postprocess_start);
            result.total_ms = elapsed_ms(total_start);
            consumer.consume(result);
            ++frame_index;
        }
        const StageTimes& times = consumer.times();
        if (times.total_ms.empty()) throw std::runtime_error("No measured frames");

        write_summary(
            std::cout, options, detector.input_height(), detector.input_width(), frame_index,
            static_cast<int>(times.total_ms.size()),
            consumer.detection_frames(), consumer.detection_count(), times, consumer.tracking_stats());
        if (!options.output_json.empty()) {
            std::ofstream summary(options.output_json, std::ios::out | std::ios::trunc);
            if (!summary) throw std::runtime_error("Unable to open output JSON");
            write_summary(
                summary, options, detector.input_height(), detector.input_width(), frame_index,
                static_cast<int>(times.total_ms.size()),
                consumer.detection_frames(), consumer.detection_count(), times, consumer.tracking_stats());
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
