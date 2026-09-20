#include "fused_half_rgb.hpp"
extern "C" int fused_half_rgb_test(const uint8_t* src,int width,int height,size_t src_stride,
                                   uint8_t* dst,size_t dst_stride) {
    return fused_half_rgb(src,width,height,src_stride,dst,dst_stride)?0:-1;
}
