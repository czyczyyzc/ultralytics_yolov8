#pragma once
#include <opencv2/core.hpp>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <time.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

inline double monotonic_ms() {
    timespec t{};
    if(clock_gettime(CLOCK_MONOTONIC,&t)) throw std::runtime_error("clock_gettime failed");
    return double(t.tv_sec)*1000.+double(t.tv_nsec)/1e6;
}
struct CaptureStamp {
    double frame_ms=0,dequeue_ms=0,copy_ms=0;
    uint32_t sequence=0,flags=0;
    bool valid=false;
};

// Explicit raw8-as-gray interpretation; never silently demosaic a color sensor.
// Frames are copied before QBUF so concurrent inference/GMC cannot race DMA.
class CameraSource {
    struct Buffer {void* data=MAP_FAILED;size_t size=0;};
    int fd=-1,w=0,h=0,stride=0;
    v4l2_buf_type type=V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    std::vector<Buffer> buffers;
    bool streaming=false,latest;
    uint32_t pixel=0;
    int call(unsigned long request,void* arg) {
        int r;do {r=ioctl(fd,request,arg);} while(r<0 && errno==EINTR);return r;
    }
    void check(unsigned long request,void* arg) {
        if(call(request,arg)<0) throw std::runtime_error("V4L2 ioctl "+std::to_string(request)+": "+std::strerror(errno));
    }
    void queue(uint32_t index) {
        v4l2_plane plane{};v4l2_buffer b{};
        b.type=type;b.memory=V4L2_MEMORY_MMAP;b.index=index;b.length=1;b.m.planes=&plane;
        check(VIDIOC_QBUF,&b);
    }
    bool dequeue(v4l2_buffer& b,v4l2_plane& p) {
        b={};p={};b.type=type;b.memory=V4L2_MEMORY_MMAP;b.length=1;b.m.planes=&p;
        if(call(VIDIOC_DQBUF,&b)==0) return true;
        if(errno==EAGAIN) return false;
        throw std::runtime_error(std::string("V4L2 DQBUF: ")+std::strerror(errno));
    }
    void cleanup() {
        if(fd<0) return;
        if(streaming) call(VIDIOC_STREAMOFF,&type);
        for(auto& b:buffers) if(b.data!=MAP_FAILED) munmap(b.data,b.size);
        close(fd);fd=-1;
    }
public:
    uint64_t discarded=0;
    double stream_start_ms=0;
    CameraSource(const std::string& path,bool newest,int n):latest(newest) {
        if(n<2 || n>16) throw std::runtime_error("Camera buffers must be 2..16");
        try {
            fd=open(path.c_str(),O_RDWR|O_NONBLOCK|O_CLOEXEC);
            if(fd<0) throw std::runtime_error(std::string("Camera open: ")+std::strerror(errno));
            v4l2_capability c{};check(VIDIOC_QUERYCAP,&c);
            uint32_t caps=(c.capabilities&V4L2_CAP_DEVICE_CAPS)?c.device_caps:c.capabilities;
            if(!(caps&V4L2_CAP_VIDEO_CAPTURE_MPLANE) || !(caps&V4L2_CAP_STREAMING))
                throw std::runtime_error("Expected streaming multiplanar capture");
            v4l2_format f{};f.type=type;check(VIDIOC_G_FMT,&f);
            auto& p=f.fmt.pix_mp;w=p.width;h=p.height;pixel=p.pixelformat;stride=p.plane_fmt[0].bytesperline;
            if(p.num_planes!=1 || (pixel!=V4L2_PIX_FMT_SBGGR8 && pixel!=V4L2_PIX_FMT_GREY) ||
               w<=0 || h<=0 || stride<w) throw std::runtime_error("Expected single-plane BA81/GREY raw8");
            v4l2_requestbuffers req{};req.type=type;req.memory=V4L2_MEMORY_MMAP;req.count=n;
            check(VIDIOC_REQBUFS,&req);
            if(req.count<2 || req.count>32) throw std::runtime_error("Invalid camera buffer count");
            buffers.resize(req.count);
            for(uint32_t i=0;i<req.count;++i) {
                v4l2_plane plane{};v4l2_buffer b{};
                b.type=type;b.memory=V4L2_MEMORY_MMAP;b.index=i;b.length=1;b.m.planes=&plane;
                check(VIDIOC_QUERYBUF,&b);buffers[i].size=plane.length;
                if(plane.length<size_t(stride)*h) throw std::runtime_error("Camera buffer too small");
                buffers[i].data=mmap(nullptr,plane.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,plane.m.mem_offset);
                if(buffers[i].data==MAP_FAILED) throw std::runtime_error("Camera mmap failed");
                queue(i);
            }
        } catch(...) {cleanup();throw;}
    }
    ~CameraSource(){cleanup();}
    CameraSource(const CameraSource&)=delete;
    CameraSource& operator=(const CameraSource&)=delete;
    int width() const{return w;}
    int height() const{return h;}
    int buffer_count() const{return buffers.size();}
    void read(cv::Mat& image,CaptureStamp& stamp) {
        if(!streaming) {stream_start_ms=monotonic_ms();check(VIDIOC_STREAMON,&type);streaming=true;}
        v4l2_plane plane{};v4l2_buffer b{};
        double deadline=monotonic_ms()+3000;
        while(!dequeue(b,plane)) {
            if(monotonic_ms()>deadline) throw std::runtime_error("Camera capture timeout");
            pollfd p{fd,POLLIN,0};int r=poll(&p,1,100);
            if(r<0 && errno!=EINTR) throw std::runtime_error("Camera poll failed");
            if(r>0 && (p.revents&(POLLERR|POLLHUP|POLLNVAL))) throw std::runtime_error("Camera disconnected");
        }
        if(latest) for(size_t i=1;i<buffers.size();++i) {
            v4l2_plane newer_plane{};v4l2_buffer newer{};
            if(!dequeue(newer,newer_plane)) break;
            queue(b.index);++discarded;b=newer;plane=newer_plane;b.m.planes=&plane;
        }
        stamp.dequeue_ms=monotonic_ms();stamp.sequence=b.sequence;stamp.flags=b.flags;
        stamp.frame_ms=double(b.timestamp.tv_sec)*1000.+double(b.timestamp.tv_usec)/1000.;
        stamp.valid=(b.flags&V4L2_BUF_FLAG_TIMESTAMP_MASK)==V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC &&
                    stamp.frame_ms>0 && stamp.frame_ms<=stamp.dequeue_ms+1;
        if(b.index>=buffers.size() || (b.flags&V4L2_BUF_FLAG_ERROR)) throw std::runtime_error("Invalid/error camera frame");
        const auto& buf=buffers[b.index];size_t offset=plane.data_offset,needed=size_t(stride)*(h-1)+w;
        if(offset>buf.size || needed>buf.size-offset || plane.bytesused<offset+needed)
            throw std::runtime_error("Truncated camera frame");
        cv::Mat raw(h,w,CV_8UC1,static_cast<unsigned char*>(buf.data)+offset,stride);
        raw.copyTo(image);stamp.copy_ms=monotonic_ms()-stamp.dequeue_ms;
        queue(b.index);
    }
};
