// Equivalent native implementation of efficient_gmc.py with optional pyramid reuse.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
struct GMC {
    int width,corners,refresh,index=0;
    bool cached,resize_first;
    cv::Mat previous,current,resized;
    std::vector<cv::Point2f> points;
    std::vector<cv::Mat> previous_pyramid,current_pyramid;
    std::array<uint64_t,4> counts{};
    std::array<double,5> seconds{};
    GMC(int w,int c,int r,bool cache,bool resize):width(w),corners(c),refresh(r),cached(cache),resize_first(resize) {
        if(w<64 || c<8 || r<1) throw std::runtime_error("Invalid GMC budget");
        cv::setNumThreads(1);
    }
    void pyramid(const cv::Mat& gray,std::vector<cv::Mat>& output) {
        auto start=Clock::now();
        cv::buildOpticalFlowPyramid(gray,output,cv::Size(21,21),3,true,
            cv::BORDER_REFLECT_101,cv::BORDER_CONSTANT,false);
        seconds[1]+=elapsed(start);
    }
    void apply(const cv::Mat& raw,double* warp) {
        auto start=Clock::now();
        int w=std::min(raw.cols,width),h=std::max(1,int(std::nearbyint(double(raw.rows)*w/raw.cols)));
        cv::Size size(w,h);
        if(resize_first) {
            if(raw.size()!=size) cv::resize(raw,resized,size,0,0,cv::INTER_LINEAR);
            else raw.copyTo(resized);
            if(raw.channels()==3) cv::cvtColor(resized,current,cv::COLOR_BGR2GRAY);
            else resized.copyTo(current);
        } else {
            if(raw.channels()==3) cv::cvtColor(raw,resized,cv::COLOR_BGR2GRAY);
            else raw.copyTo(resized);
            if(resized.size()!=size) cv::resize(resized,current,size,0,0,cv::INTER_LINEAR);
            else resized.copyTo(current);
        }
        seconds[0]+=elapsed(start);
        const double identity[]={1,0,0,0,1,0};std::copy(identity,identity+6,warp);
        bool accepted=false,built=false;
        std::vector<cv::Point2f> next_points;
        if(!previous.empty() && previous.size()==current.size() && points.size()>=6) {
            std::vector<cv::Point2f> moved;
            std::vector<uchar> status;
            std::vector<float> errors;
            if(cached) {
                if(previous_pyramid.empty()) pyramid(previous,previous_pyramid);
                pyramid(current,current_pyramid);built=true;
            }
            start=Clock::now();
            auto criteria=cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,.01);
            if(cached) cv::calcOpticalFlowPyrLK(previous_pyramid,current_pyramid,points,moved,status,errors,cv::Size(21,21),3,criteria);
            else cv::calcOpticalFlowPyrLK(previous,current,points,moved,status,errors,cv::Size(21,21),3,criteria);
            seconds[2]+=elapsed(start);
            std::vector<cv::Point2f> before,after;
            for(size_t i=0;i<moved.size();++i) {
                auto p=moved[i];
                if(status[i] && std::isfinite(p.x) && std::isfinite(p.y) && p.x>=0 && p.x<w && p.y>=0 && p.y<h) {
                    before.push_back(points[i]);after.push_back(p);
                }
            }
            if(after.size()>=6) {
                start=Clock::now();cv::Mat inliers;
                auto small=cv::estimateAffinePartial2D(before,after,inliers,cv::RANSAC,6.*w/raw.cols,2000,.99,10);
                seconds[3]+=elapsed(start);
                if(!small.empty() && cv::checkRange(small) && !inliers.empty()) {
                    int n=cv::countNonZero(inliers);
                    double scale=std::hypot(small.at<double>(0,0),small.at<double>(1,0));
                    if(n>=6 && double(n)/after.size()>=.3 && scale>=.5 && scale<=2) {
                        double sx=double(raw.cols)/w,sy=double(raw.rows)/h;
                        warp[0]=(sx*small.at<double>(0,0))*(1/sx);
                        warp[1]=(sx*small.at<double>(0,1))*(1/sy);
                        warp[2]=small.at<double>(0,2)*sx;
                        warp[3]=(sy*small.at<double>(1,0))*(1/sx);
                        warp[4]=(sy*small.at<double>(1,1))*(1/sy);
                        warp[5]=small.at<double>(1,2)*sy;
                        for(size_t i=0;i<after.size();++i) if(inliers.at<uchar>(int(i))) next_points.push_back(after[i]);
                        accepted=true;
                    }
                }
            }
        }
        if(accepted) ++counts[1];else if(index) ++counts[2];
        if(index%refresh==0 || next_points.size()<size_t(std::max(6,corners/2))) {
            start=Clock::now();
            cv::goodFeaturesToTrack(current,next_points,corners,.01,4,cv::noArray(),3,false,.04);
            seconds[4]+=elapsed(start);++counts[3];
        }
        if(cached) {
            if(!built && next_points.size()>=6) {pyramid(current,current_pyramid);built=true;}
            if(built) previous_pyramid.swap(current_pyramid);
            else previous_pyramid.clear();
        }
        std::swap(previous,current);points.swap(next_points);++index;++counts[0];
    }
};
thread_local std::string error;
}
extern "C" {
const char* gmc_error() { return error.c_str(); }
void* gmc_create(int width,int corners,int refresh,int cached,int resize_first) {
    try { return new GMC(width,corners,refresh,cached,resize_first); }
    catch(const std::exception& e) {error=e.what();return nullptr;}
}
void gmc_destroy(void* p) { delete static_cast<GMC*>(p); }
int gmc_apply(void* p,unsigned char* data,int width,int height,size_t stride,int channels,double* warp) {
    try {
        if(!p || !data || width<=0 || height<=0 || (channels!=1 && channels!=3) || stride<size_t(width*channels) || !warp)
            throw std::runtime_error("Invalid GMC image");
        static_cast<GMC*>(p)->apply(cv::Mat(height,width,CV_MAKETYPE(CV_8U,channels),data,stride),warp);
        return 0;
    } catch(const std::exception& e) {error=e.what();return -1;}
}
void gmc_stats(void* p,uint64_t* counts,double* seconds) {
    auto& gmc=*static_cast<GMC*>(p);
    std::copy(gmc.counts.begin(),gmc.counts.end(),counts);
    std::copy(gmc.seconds.begin(),gmc.seconds.end(),seconds);
}
}
