#include "video_source.hpp"
#include <iostream>
#include <iomanip>

// Isolate decoder/FFmpeg issues without loading RKNN or allocating NPU buffers.
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("Usage: decode_probe VIDEO opencv|ffmpeg|rkmpp FRAMES");
        cv::setNumThreads(1);
        std::cerr<<"opening "<<argv[2]<<std::endl;
        VideoSource source(argv[1],argv[2]);
        std::cerr<<"opened "<<source.width()<<'x'<<source.height()<<std::endl;
        int count=std::stoi(argv[3]);if(count<=0) throw std::runtime_error("Invalid frame count");
        auto start=std::chrono::steady_clock::now();
        for(int i=0;i<count;++i) {
            cv::Mat image;
            if(!source.read(image)) throw std::runtime_error("Early EOF");
            auto mean=cv::mean(image);
            std::cout<<i<<' '<<std::setprecision(10)<<mean[0]<<' '<<mean[1]<<' '<<mean[2]<<std::endl;
        }
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cerr<<"decode_only_fps="<<count/seconds<<std::endl;
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
