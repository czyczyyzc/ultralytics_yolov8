#include "fused_half_rgb.hpp"
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

int main() {
    std::mt19937 rng(20260920);int cases=0;
    for(int w:{2,4,14,16,18,30,32,34,62,128,1920})
    for(int h:{2,6,18,1080}) for(int pad:{0,1,17}) {
        size_t ss=size_t(w)*3+pad,ds=size_t(w/2)*3+pad;
        std::vector<uint8_t> source(ss*h),actual(ds*(h/2),157),expected=actual;
        for(auto& v:source) v=rng()%256;
        for(int y=0;y<h/2;++y) for(int x=0;x<w/2;++x) for(int c=0;c<3;++c) {
            unsigned value=0;
            for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx)
                value+=source[size_t(y*2+dy)*ss+(x*2+dx)*3+c];
            expected[size_t(y)*ds+x*3+2-c]=(value+2)/4;
        }
        if(!fused_half_rgb(source.data(),w,h,ss,actual.data(),ds) || actual!=expected)
            throw std::runtime_error("Pixel or padding mismatch");
        ++cases;
    }
    uint8_t data[24]{};
    if(fused_half_rgb(nullptr,2,2,6,data,3) || fused_half_rgb(data,3,2,9,data,3) ||
       fused_half_rgb(data,2,3,6,data,3) || fused_half_rgb(data,2,2,5,data,3) ||
       fused_half_rgb(data,2,2,6,data,2)) throw std::runtime_error("Invalid input accepted");
    std::cout<<"passed "<<cases<<" randomized layouts and 5 invalid inputs\n";
}
