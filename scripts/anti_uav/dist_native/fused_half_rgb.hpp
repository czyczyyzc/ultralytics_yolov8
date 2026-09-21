#pragma once
#include <cstddef>
#include <cstdint>
#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// Exact 2x reduction: four uint8 samples, round-half-up, then BGR -> RGB.
// Source/destination must not overlap. Strides may include padding.
inline bool fused_half_rgb(const uint8_t* src,int width,int height,size_t src_stride,
                           uint8_t* dst,size_t dst_stride) {
    if(!src || !dst || width<2 || height<2 || width%2 || height%2 ||
       src_stride<size_t(width)*3 || dst_stride<size_t(width/2)*3) return false;
    const int output_w=width/2;
    for(int y=0;y<height/2;++y) {
        const uint8_t* a=src+size_t(2*y)*src_stride;
        const uint8_t* b=a+src_stride;
        uint8_t* out=dst+size_t(y)*dst_stride;
        int x=0;
#if defined(__aarch64__) || defined(__ARM_NEON)
        for(;x+8<=output_w;x+=8) {
            auto top=vld3q_u8(a+size_t(x)*6),bottom=vld3q_u8(b+size_t(x)*6);
            uint8x8x3_t rgb;
            for(int c=0;c<3;++c) {
                auto sum=vaddq_u16(vpaddlq_u8(top.val[c]),vpaddlq_u8(bottom.val[c]));
                rgb.val[2-c]=vshrn_n_u16(vaddq_u16(sum,vdupq_n_u16(2)),2);
            }
            vst3_u8(out+size_t(x)*3,rgb);
        }
#endif
        for(;x<output_w;++x) for(int c=0;c<3;++c) {
            size_t i=size_t(x)*6+c;
            out[size_t(x)*3+2-c]=uint8_t((unsigned(a[i])+a[i+3]+b[i]+b[i+3]+2)>>2);
        }
    }
    return true;
}

// Monochrome raw8: resize once, then replicate into the model's RGB channels.
inline bool fused_half_gray_rgb(const uint8_t* src,int width,int height,size_t ss,
                                uint8_t* dst,size_t ds) {
    if(!src || !dst || width<2 || height<2 || width%2 || height%2 ||
       ss<size_t(width) || ds<size_t(width/2)*3) return false;
    for(int y=0;y<height/2;++y) {
        const auto* a=src+size_t(y*2)*ss;const auto* b=a+ss;
        auto* out=dst+size_t(y)*ds;int x=0;
#if defined(__aarch64__) || defined(__ARM_NEON)
        for(;x+8<=width/2;x+=8) {
            auto sum=vaddq_u16(vpaddlq_u8(vld1q_u8(a+x*2)),vpaddlq_u8(vld1q_u8(b+x*2)));
            auto value=vshrn_n_u16(vaddq_u16(sum,vdupq_n_u16(2)),2);
            uint8x8x3_t rgb{{value,value,value}};vst3_u8(out+x*3,rgb);
        }
#endif
        for(;x<width/2;++x) {
            auto value=uint8_t((unsigned(a[x*2])+a[x*2+1]+b[x*2]+b[x*2+1]+2)>>2);
            out[x*3]=out[x*3+1]=out[x*3+2]=value;
        }
    }
    return true;
}
