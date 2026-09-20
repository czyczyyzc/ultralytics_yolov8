// Reuse the native INT8 decoder; the CLI and its tracker are never invoked.
#define main native_detector_unused_main
#include "native_yolov8_video.cpp"
#undef main
#include <type_traits>
#ifdef AU_ENABLE_RGA
#include <rga/im2d.h>
#endif

namespace {
thread_local std::string api_error;
template <typename T> T* make_detector(const char* model, const char* core, bool defer_io = false) {
    if constexpr (std::is_constructible_v<T, const std::string&, const std::string&,
                                        bool, const std::string&, bool>) {
        return new T(model, core, false, "opencv", defer_io);
    } else {
        return new T(model, core, defer_io);
    }
}
}

extern "C" {
int au_detector_rga_supported() {
#ifdef AU_ENABLE_RGA
    return 1;
#else
    return 0;
#endif
}
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
int au_detector_create_pool(const char* model, const char** cores, int count,
                            int threads, void** handles) {
    try {
        if (count < 1 || count > 3 || !handles || !cores)
            throw std::runtime_error("Invalid RKNN pool size");
        cv::setNumThreads(threads);
        std::vector<std::unique_ptr<NativeYoloV8>> pool;
        pool.emplace_back(make_detector<NativeYoloV8>(model, cores[0], true));
        for (int i = 1; i < count; ++i)
            pool.emplace_back(new NativeYoloV8(*pool.front(), cores[i], true));
        // Duplicate model/weights before binding separate per-worker I/O buffers.
        for (int i = 0; i < count; ++i) {
            pool[i]->initialize_deferred_io(cores[i]);
            if (pool[i]->input_width() != 960 || pool[i]->input_height() != 544)
                throw std::runtime_error("This deployment requires 960x544 input");
            pool[i]->set_padding_value(114);
        }
        for (int i = 0; i < count; ++i) handles[i] = pool[i].release();
        return 0;
    } catch (const std::exception& e) {
        api_error = e.what();
        return -1;
    }
}
int au_detector_infer_ex(void* handle, unsigned char* bgr, int width, int height,
                      size_t stride, float conf, float iou, int capacity,
                      float* boxes, double* times_ms, int cached_preprocess) {
    try {
        if (!handle || !bgr || width <= 0 || height <= 0 || capacity <= 0 ||
            stride < static_cast<size_t>(width) * 3 || !boxes || !times_ms)
            throw std::runtime_error("Invalid detector input");
        auto& detector = *static_cast<NativeYoloV8*>(handle);
        cv::Mat frame(height, width, CV_8UC3, bgr, stride);
        auto start = Clock::now();
        LetterboxInfo letterbox;
        if (cached_preprocess) {
            // OpenCV operates in cached RAM; only the final bulk copy touches DMA memory.
            thread_local cv::Mat resized, rgb;
            const float ratio = std::min(static_cast<float>(detector.input_height()) / height,
                                         static_cast<float>(detector.input_width()) / width);
            const int rw = std::max(1, static_cast<int>(std::round(width * ratio)));
            const int rh = std::max(1, static_cast<int>(std::round(height * ratio)));
            const float dw = (detector.input_width() - rw) * .5f;
            const float dh = (detector.input_height() - rh) * .5f;
            rgb.create(detector.input_height(), detector.input_width(), CV_8UC3);
            rgb.setTo(cv::Scalar::all(114));
            if(cached_preprocess==2) {
#ifdef AU_ENABLE_RGA
                if(frame.step%3) throw std::runtime_error("RGA requires a whole-pixel stride");
                resized.create(rh,rw,CV_8UC3);
                auto src=wrapbuffer_virtualaddr(frame.data,width,height,RK_FORMAT_BGR_888,int(frame.step/3),height);
                auto dst=wrapbuffer_virtualaddr(resized.data,rw,rh,RK_FORMAT_BGR_888,int(resized.step/3),rh);
                auto status=imresize(src,dst,0,0,1,1);
                if(status!=IM_STATUS_SUCCESS) throw std::runtime_error(std::string("RGA resize: ")+imStrError(status));
#else
                throw std::runtime_error("RGA preprocessing not compiled");
#endif
            } else cv::resize(frame, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
            cv::Mat roi = rgb(cv::Rect(static_cast<int>(std::round(dw-.1f)),
                                      static_cast<int>(std::round(dh-.1f)), rw, rh));
            cv::cvtColor(resized, roi, cv::COLOR_BGR2RGB);
            detector.copy_rgb_input(rgb);
            letterbox = LetterboxInfo{ratio, dw, dh};
        } else {
            letterbox = detector.preprocess(frame);
        }
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
int au_detector_infer(void* handle, unsigned char* bgr, int width, int height,
                      size_t stride, float conf, float iou, int capacity,
                      float* boxes, double* times_ms) {
    return au_detector_infer_ex(handle, bgr, width, height, stride, conf, iou,
                                capacity, boxes, times_ms, 0);
}
}
