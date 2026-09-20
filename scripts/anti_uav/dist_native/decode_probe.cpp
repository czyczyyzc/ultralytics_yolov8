#include "video_source.hpp"
#include <iostream>
#include <iomanip>

// Isolate decoder/FFmpeg issues without loading RKNN or allocating NPU buffers.
int main(int argc,char** argv) {
    try {
        if(argc<4 || argc>6) throw std::runtime_error("Usage: decode_probe VIDEO opencv|ffmpeg|rkmpp FRAMES [THREADS [slice|frame|auto]]");
        cv::setNumThreads(1);
        std::cerr<<"opening "<<argv[2]<<std::endl;
        VideoSource source(argv[1],argv[2],argc>4?std::stoi(argv[4]):1,argc>5?argv[5]:"slice");
        std::cerr<<"opened "<<source.width()<<'x'<<source.height()<<std::endl;
        int count=std::stoi(argv[3]);if(count<=0) throw std::runtime_error("Invalid frame count");
        auto start=std::chrono::steady_clock::now();
        double first_ms=0;
        for(int i=0;i<count;++i) {
            cv::Mat image;
            if(!source.read(image)) throw std::runtime_error("Early EOF");
            if(!i) first_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            auto mean=cv::mean(image);
            std::cout<<i<<' '<<std::setprecision(10)<<mean[0]<<' '<<mean[1]<<' '<<mean[2]<<std::endl;
        }
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cerr<<"decode_only_fps="<<count/seconds<<" first_read_ms="<<first_ms<<std::endl;
    } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
