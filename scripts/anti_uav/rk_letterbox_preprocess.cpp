// RK3588 detector letterbox preprocess helper.
//
// The fast path uses Rockchip RGA to resize BGR input directly into the RGB
// letterbox ROI. If RGA rejects a stride/format combination, OpenCV C++ is used
// as a safe fallback with the same output contract.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef WITH_RGA
#include <rga/im2d.h>
#endif

#ifdef WITH_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace {

enum BackendUsed {
    BACKEND_NONE = 0,
    BACKEND_RGA = 1,
    BACKEND_OPENCV = 2,
};

void fill_rgb(uint8_t* dst, int height, int width, int pad_r, int pad_g, int pad_b) {
    const int total = height * width;
    if (pad_r == 0 && pad_g == 0 && pad_b == 0) {
        std::memset(dst, 0, static_cast<size_t>(total) * 3);
        return;
    }
    for (int index = 0; index < total; ++index) {
        uint8_t* pixel = dst + index * 3;
        pixel[0] = static_cast<uint8_t>(pad_r);
        pixel[1] = static_cast<uint8_t>(pad_g);
        pixel[2] = static_cast<uint8_t>(pad_b);
    }
}

void copy_bgr_to_rgb_roi(
    const uint8_t* src_bgr,
    int src_h,
    int src_w,
    int src_stride_bytes,
    uint8_t* dst_rgb,
    int dst_w,
    int top,
    int left
) {
    for (int y = 0; y < src_h; ++y) {
        const uint8_t* src_row = src_bgr + static_cast<size_t>(y) * src_stride_bytes;
        uint8_t* dst_row = dst_rgb + (static_cast<size_t>(top + y) * dst_w + left) * 3;
        for (int x = 0; x < src_w; ++x) {
            const uint8_t* src_px = src_row + x * 3;
            uint8_t* dst_px = dst_row + x * 3;
            dst_px[0] = src_px[2];
            dst_px[1] = src_px[1];
            dst_px[2] = src_px[0];
        }
    }
}

#ifdef WITH_RGA
bool try_rga_resize_bgr_to_rgb(
    const uint8_t* src,
    int src_h,
    int src_w,
    int src_stride_bytes,
    uint8_t* dst,
    int dst_h,
    int dst_w,
    int top,
    int left,
    int resized_h,
    int resized_w
) {
    if (src_stride_bytes <= 0 || src_stride_bytes % 3 != 0) {
        return false;
    }
    const int src_wstride = src_stride_bytes / 3;
    // RGA color conversion between BGR_888/RGB_888 can differ from OpenCV's
    // channel semantics. Keep RGA responsible only for resize, then perform the
    // BGR->RGB channel swap explicitly before writing into the letterbox ROI.
    thread_local std::vector<uint8_t> resized_bgr;
    resized_bgr.resize(static_cast<size_t>(resized_h) * resized_w * 3);
    rga_buffer_t src_buffer = wrapbuffer_virtualaddr(
        const_cast<uint8_t*>(src),
        src_w,
        src_h,
        RK_FORMAT_BGR_888,
        src_wstride,
        src_h
    );
    rga_buffer_t dst_buffer = wrapbuffer_virtualaddr(
        resized_bgr.data(),
        resized_w,
        resized_h,
        RK_FORMAT_BGR_888,
        resized_w,
        resized_h
    );
    const IM_STATUS status = imresize(src_buffer, dst_buffer, 0, 0, INTER_LINEAR, 1);
    if (status != IM_STATUS_SUCCESS) {
        return false;
    }
    copy_bgr_to_rgb_roi(resized_bgr.data(), resized_h, resized_w, resized_w * 3, dst, dst_w, top, left);
    return true;
}
#endif

#ifdef WITH_OPENCV
bool try_opencv_resize_bgr_to_rgb(
    const uint8_t* src,
    int src_h,
    int src_w,
    int src_stride_bytes,
    uint8_t* dst,
    int dst_h,
    int dst_w,
    int top,
    int left,
    int resized_h,
    int resized_w
) {
    cv::Mat src_mat(src_h, src_w, CV_8UC3, const_cast<uint8_t*>(src), static_cast<size_t>(src_stride_bytes));
    cv::Mat dst_mat(dst_h, dst_w, CV_8UC3, dst);
    cv::Mat roi = dst_mat(cv::Rect(left, top, resized_w, resized_h));
    if (src_w == resized_w && src_h == resized_h) {
        cv::cvtColor(src_mat, roi, cv::COLOR_BGR2RGB);
        return true;
    }
    cv::Mat resized;
    cv::resize(src_mat, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);
    cv::cvtColor(resized, roi, cv::COLOR_BGR2RGB);
    return true;
}
#endif

}  // namespace

extern "C" int rk_letterbox_preprocess_has_rga() {
#ifdef WITH_RGA
    return 1;
#else
    return 0;
#endif
}

extern "C" int rk_letterbox_preprocess_has_opencv() {
#ifdef WITH_OPENCV
    return 1;
#else
    return 0;
#endif
}

extern "C" int rk_letterbox_preprocess_bgr_to_rgb_u8(
    const uint8_t* src,
    int src_h,
    int src_w,
    int src_stride_bytes,
    uint8_t* dst,
    int dst_h,
    int dst_w,
    int pad_r,
    int pad_g,
    int pad_b,
    int prefer_rga,
    float* out_ratio,
    float* out_dw,
    float* out_dh,
    int* out_backend
) {
    if (src == nullptr || dst == nullptr || src_h <= 0 || src_w <= 0 || dst_h <= 0 || dst_w <= 0) {
        return -1;
    }
    if (src_stride_bytes < src_w * 3) {
        return -2;
    }

    const float ratio = std::min(
        static_cast<float>(dst_h) / static_cast<float>(src_h),
        static_cast<float>(dst_w) / static_cast<float>(src_w)
    );
    const int resized_w = std::max(1, static_cast<int>(std::round(static_cast<float>(src_w) * ratio)));
    const int resized_h = std::max(1, static_cast<int>(std::round(static_cast<float>(src_h) * ratio)));
    const float dw = (static_cast<float>(dst_w - resized_w)) * 0.5f;
    const float dh = (static_cast<float>(dst_h - resized_h)) * 0.5f;
    const int left = static_cast<int>(std::round(dw - 0.1f));
    const int top = static_cast<int>(std::round(dh - 0.1f));

    if (out_ratio != nullptr) {
        *out_ratio = ratio;
    }
    if (out_dw != nullptr) {
        *out_dw = dw;
    }
    if (out_dh != nullptr) {
        *out_dh = dh;
    }
    if (out_backend != nullptr) {
        *out_backend = BACKEND_NONE;
    }

    fill_rgb(dst, dst_h, dst_w, pad_r, pad_g, pad_b);

#ifdef WITH_RGA
    if (prefer_rga != 0
        && try_rga_resize_bgr_to_rgb(src, src_h, src_w, src_stride_bytes, dst, dst_h, dst_w, top, left, resized_h, resized_w)) {
        if (out_backend != nullptr) {
            *out_backend = BACKEND_RGA;
        }
        return 0;
    }
#endif

#ifdef WITH_OPENCV
    if (try_opencv_resize_bgr_to_rgb(src, src_h, src_w, src_stride_bytes, dst, dst_h, dst_w, top, left, resized_h, resized_w)) {
        if (out_backend != nullptr) {
            *out_backend = BACKEND_OPENCV;
        }
        return 0;
    }
#endif

    return -3;
}
