// Minimal YOLOv8 RKOPT postprocess path aligned with rknn_model_zoo's C++ flow.
//
// Inputs are dequantized NCHW float tensors from RKNNLite:
//   per scale: bbox logits [64,H,W], class scores [C,H,W], score_sum [H,W].
// The implementation filters grid cells before DFL, then runs classwise NMS.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

struct Candidate {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int cls;
};

float iou_xyxy(const Candidate& a, const Candidate& b) {
    const float inter_x1 = std::max(a.x1, b.x1);
    const float inter_y1 = std::max(a.y1, b.y1);
    const float inter_x2 = std::min(a.x2, b.x2);
    const float inter_y2 = std::min(a.y2, b.y2);
    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter = inter_w * inter_h;
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float denom = area_a + area_b - inter;
    return denom <= 1e-6f ? 0.0f : inter / denom;
}

void compute_dfl(const float* box, int grid_offset, int grid_len, int dfl_len, float out[4]) {
    for (int side = 0; side < 4; ++side) {
        const int channel_base = side * dfl_len;
        float max_value = box[(channel_base * grid_len) + grid_offset];
        for (int k = 1; k < dfl_len; ++k) {
            max_value = std::max(max_value, box[((channel_base + k) * grid_len) + grid_offset]);
        }
        float exp_sum = 0.0f;
        float weighted_sum = 0.0f;
        for (int k = 0; k < dfl_len; ++k) {
            const float value = std::exp(box[((channel_base + k) * grid_len) + grid_offset] - max_value);
            exp_sum += value;
            weighted_sum += value * static_cast<float>(k);
        }
        out[side] = exp_sum <= 1e-12f ? 0.0f : weighted_sum / exp_sum;
    }
}

void process_branch(
    const float* box,
    const float* cls,
    const float* score_sum,
    int box_c,
    int cls_c,
    int grid_h,
    int grid_w,
    int input_h,
    int input_w,
    float conf,
    std::vector<Candidate>& candidates
) {
    if (box == nullptr || cls == nullptr || box_c <= 0 || cls_c <= 0 || grid_h <= 0 || grid_w <= 0) {
        return;
    }
    const int grid_len = grid_h * grid_w;
    const int dfl_len = box_c / 4;
    if (dfl_len <= 0) {
        return;
    }
    const float stride_x = static_cast<float>(input_w) / static_cast<float>(grid_w);
    const float stride_y = static_cast<float>(input_h) / static_cast<float>(grid_h);

    for (int y = 0; y < grid_h; ++y) {
        for (int x = 0; x < grid_w; ++x) {
            const int offset = y * grid_w + x;
            if (score_sum != nullptr && score_sum[offset] < conf) {
                continue;
            }

            int best_cls = -1;
            float best_score = conf;
            for (int c = 0; c < cls_c; ++c) {
                const float score = cls[c * grid_len + offset];
                if (score >= conf && score > best_score) {
                    best_score = score;
                    best_cls = c;
                }
            }
            if (best_cls < 0) {
                continue;
            }

            float dist[4];
            compute_dfl(box, offset, grid_len, dfl_len, dist);
            const float cx = static_cast<float>(x) + 0.5f;
            const float cy = static_cast<float>(y) + 0.5f;
            candidates.push_back(
                Candidate{
                    (cx - dist[0]) * stride_x,
                    (cy - dist[1]) * stride_y,
                    (cx + dist[2]) * stride_x,
                    (cy + dist[3]) * stride_y,
                    best_score,
                    best_cls,
                }
            );
        }
    }
}

}  // namespace

extern "C" int rk_yolov8_postprocess_float(
    const float* box0,
    const float* cls0,
    const float* score0,
    int box_c0,
    int cls_c0,
    int h0,
    int w0,
    const float* box1,
    const float* cls1,
    const float* score1,
    int box_c1,
    int cls_c1,
    int h1,
    int w1,
    const float* box2,
    const float* cls2,
    const float* score2,
    int box_c2,
    int cls_c2,
    int h2,
    int w2,
    int input_h,
    int input_w,
    float conf,
    float nms_thresh,
    int max_det,
    float* out_boxes,
    int* out_classes,
    float* out_scores
) {
    if (out_boxes == nullptr || out_classes == nullptr || out_scores == nullptr) {
        return -1;
    }

    std::vector<Candidate> candidates;
    candidates.reserve(512);
    process_branch(box0, cls0, score0, box_c0, cls_c0, h0, w0, input_h, input_w, conf, candidates);
    process_branch(box1, cls1, score1, box_c1, cls_c1, h1, w1, input_h, input_w, conf, candidates);
    process_branch(box2, cls2, score2, box_c2, cls_c2, h2, w2, input_h, input_w, conf, candidates);
    if (candidates.empty()) {
        return 0;
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.score > b.score;
    });

    std::vector<Candidate> kept;
    kept.reserve(max_det > 0 ? max_det : 128);
    std::vector<uint8_t> suppressed(candidates.size(), 0);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        kept.push_back(candidates[i]);
        if (max_det > 0 && static_cast<int>(kept.size()) >= max_det) {
            break;
        }
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j] || candidates[i].cls != candidates[j].cls) {
                continue;
            }
            if (iou_xyxy(candidates[i], candidates[j]) > nms_thresh) {
                suppressed[j] = 1;
            }
        }
    }

    for (size_t i = 0; i < kept.size(); ++i) {
        out_boxes[i * 4 + 0] = kept[i].x1;
        out_boxes[i * 4 + 1] = kept[i].y1;
        out_boxes[i * 4 + 2] = kept[i].x2;
        out_boxes[i * 4 + 3] = kept[i].y2;
        out_classes[i] = kept[i].cls;
        out_scores[i] = kept[i].score;
    }
    return static_cast<int>(kept.size());
}
