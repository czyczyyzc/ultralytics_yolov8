// SPDX-License-Identifier: AGPL-3.0-only
// Cached-measurement C++ kernel timing only, never camera/RKNN end-to-end latency.
#include "motion_tracker.cpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

struct Frame {
    double time,quality;
    std::array<double,6> warp;
    std::vector<float> boxes;
};
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("Usage: motion_benchmark config.txt frames.txt repeats");
        std::ifstream cfg(argv[1]),input(argv[2]);int nc=0,count=0;
        cfg>>nc;if(nc!=14 && nc!=16) throw std::runtime_error("Invalid config length");
        std::vector<double> config(nc);for(auto& v:config) cfg>>v;
        if(!cfg) throw std::runtime_error("Truncated configuration");
        input>>count;if(count<=0) throw std::runtime_error("Empty measurement input");
        std::vector<Frame> frames(count);size_t capacity=1;
        for(auto& f:frames) {
            int n=0;input>>f.time>>f.quality;for(auto& v:f.warp) input>>v;input>>n;
            if(n<0 || n>10000) throw std::runtime_error("Invalid detection count");
            f.boxes.resize(5*n);for(auto& v:f.boxes) input>>v;
            if(!input) throw std::runtime_error("Truncated measurement input");
            capacity=std::max(capacity,size_t(n));
        }
        std::string extra;if(input>>extra) throw std::runtime_error("Unexpected trailing measurements");
        int repeats=std::stoi(argv[3]);if(repeats<1 || repeats>100) throw std::runtime_error("Invalid repeats");
        std::vector<int> pairs(2*capacity),triples(3*capacity);
        std::vector<double> times;times.reserve(frames.size()*repeats);
        uint64_t association_counts[8]{};size_t emitted=0,confirmed=0;
        for(int repeat=0;repeat<repeats;++repeat) {
            void* tracker=motion_create(config.data(),nc);if(!tracker) throw std::runtime_error(motion_error());
            for(const auto& f:frames) {
                auto start=std::chrono::steady_clock::now();
                int n=motion_update(tracker,f.boxes.data(),f.boxes.size()/5,f.warp.data(),f.quality,f.time,pairs.data(),capacity);
                int o=motion_observations(tracker,triples.data(),capacity);
                auto finish=std::chrono::steady_clock::now();
                if(n<0 || o<0) {motion_destroy(tracker);throw std::runtime_error(motion_error());}
                if(o!=int(f.boxes.size()/5)) throw std::runtime_error("Measurement output loss");
                times.push_back(std::chrono::duration<double,std::milli>(finish-start).count());
                emitted+=o;confirmed+=n;
            }
            motion_assignment_stats(tracker,association_counts);motion_destroy(tracker);
        }
        double sum=std::accumulate(times.begin(),times.end(),0.);std::sort(times.begin(),times.end());
        auto percentile=[&](double p) {return times[size_t(std::ceil(p*times.size()))-1];};
        std::cout<<std::setprecision(17)<<"{\"scope\":\"Native association + observation ABI on cached measurements; excludes detector, GMC, camera, decoding, and exposure\",\"frames\":"<<count<<",\"repeats\":"<<repeats
            <<",\"samples\":"<<times.size()<<",\"emitted_boxes_all_repeats\":"<<emitted<<",\"confirmed_boxes_all_repeats\":"<<confirmed
            <<",\"milliseconds\":{\"mean\":"<<sum/times.size()<<",\"p50\":"<<percentile(.5)<<",\"p95\":"<<percentile(.95)<<",\"p99\":"<<percentile(.99)<<",\"max\":"<<times.back()<<"},\"association_counts_last_repeat\":[";
        for(int i=0;i<8;++i) {if(i) std::cout<<',';std::cout<<association_counts[i];}
        std::cout<<"]}\n";return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
