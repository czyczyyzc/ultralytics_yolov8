// Reuse the native INT8 decoder; the CLI and its tracker are never invoked.
#define main native_detector_unused_main
#include "native_yolov8_video.cpp"
#undef main
#include <type_traits>

namespace {
thread_local std::string api_error;
template <typename T> T* make_detector(const char* model, const char* core) {
    if constexpr (std::is_constructible_v<T, const std::string&, const std::string&,
                                        bool, const std::string&, bool>) {
        return new T(model, core, false, "opencv", false);
    } else {
        return new T(model, core);
    }
}
}

extern "C" {
const char* au_detector_error() { return api_error.c_str(); }
void* au_detector_create(const char* model, const char* core, int threads) {
    try {
        cv::setNumThreads(threads);
        std::unique_ptr<NativeYoloV8> detector(make_detector<NativeYoloV8>(model, core));
        if (detector->input_width() != 960 || detector->input_height() != 544)
            throw std::runtime_error("This deployment requires 960x544 input");
        detector->set_padding_value(114);
        return detector.release();
    } catch (const std::exception& e) {
        api_error = e.what();
        return nullptr;
    }
}
void au_detector_destroy(void* handle) { delete static_cast<NativeYoloV8*>(handle); }
int au_detector_infer(void* handle, unsigned char* bgr, int width, int height,
                      size_t stride, float conf, float iou, int capacity,
                      float* boxes, double* times_ms) {
    try {
        if (!handle || !bgr || width <= 0 || height <= 0 || capacity <= 0 ||
            stride < static_cast<size_t>(width) * 3 || !boxes || !times_ms)
            throw std::runtime_error("Invalid detector input");
        auto& detector = *static_cast<NativeYoloV8*>(handle);
        cv::Mat frame(height, width, CV_8UC3, bgr, stride);
        auto start = Clock::now();
        const auto letterbox = detector.preprocess(frame);
        times_ms[0] = elapsed_ms(start);
        start = Clock::now();
        detector.run();
        times_ms[1] = elapsed_ms(start);
        start = Clock::now();
        const auto found = detector.decode(letterbox, width, height, conf, iou, capacity);
        times_ms[2] = elapsed_ms(start);
        for (size_t i = 0; i < found.size(); ++i) {
            const auto& d = found[i];
            boxes[5*i] = d.x1; boxes[5*i+1] = d.y1;
            boxes[5*i+2] = d.x2; boxes[5*i+3] = d.y2; boxes[5*i+4] = d.score;
        }
        return static_cast<int>(found.size());
    } catch (const std::exception& e) {
        api_error = e.what();
        return -1;
    }
}
}
