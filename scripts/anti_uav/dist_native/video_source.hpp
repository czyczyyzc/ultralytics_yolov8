#pragma once
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#ifdef WITH_MPP_SOURCE
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext_drm.h>
#include <libswscale/swscale.h>
}
#include <drm_fourcc.h>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <cerrno>
#endif

// Hardware decoding is explicit and fail-closed: no silent CPU fallback.
class VideoSource {
    cv::VideoCapture cpu;
    std::string backend;
    double rate=0;
    int count=0,w=0,h=0;
#ifdef WITH_MPP_SOURCE
    AVFormatContext* format=nullptr;
    AVCodecContext* codec=nullptr;
    AVPacket* packet=nullptr;
    AVFrame* frame=nullptr;
    SwsContext* scale=nullptr;
    int stream=-1;
    bool pending=false,eof=false,flushed=false;
    void cleanup() {
        sws_freeContext(scale);scale=nullptr;
        av_frame_free(&frame);av_packet_free(&packet);
        avcodec_free_context(&codec);avformat_close_input(&format);
    }
    static void check(int status,const char* action) {
        if(status>=0) return;
        char error[AV_ERROR_MAX_STRING_SIZE];av_strerror(status,error,sizeof(error));
        throw std::runtime_error(std::string(action)+": "+error);
    }
    struct Mapping {
        int fd;
        void* data=MAP_FAILED;
        size_t size;
        bool synced=false;
        Mapping(int f,size_t n):fd(f),size(n) {
            dma_buf_sync sync{DMA_BUF_SYNC_START|DMA_BUF_SYNC_READ};
            if(ioctl(fd,DMA_BUF_IOCTL_SYNC,&sync)<0 && errno!=ENOTTY)
                throw std::runtime_error("DMA read sync failed");
            synced=true;data=mmap(nullptr,size,PROT_READ,MAP_SHARED,fd,0);
            if(data==MAP_FAILED) {finish();throw std::runtime_error("DMA mmap failed");}
        }
        void finish() {
            if(data!=MAP_FAILED) munmap(data,size);
            if(synced) {dma_buf_sync sync{DMA_BUF_SYNC_END|DMA_BUF_SYNC_READ};ioctl(fd,DMA_BUF_IOCTL_SYNC,&sync);}
        }
        ~Mapping(){finish();}
    };
    void convert(cv::Mat& image) {
        if(frame->width!=w || frame->height!=h) throw std::runtime_error("Video dimensions changed");
        const uint8_t* src[4]={frame->data[0],frame->data[1],frame->data[2],nullptr};
        int stride[4]={frame->linesize[0],frame->linesize[1],frame->linesize[2],0};
        AVPixelFormat pixel=AVPixelFormat(frame->format);
        std::unique_ptr<Mapping> mapping;
        if(pixel==AV_PIX_FMT_DRM_PRIME) {
            auto* desc=reinterpret_cast<const AVDRMFrameDescriptor*>(frame->data[0]);
            if(!desc || desc->nb_objects!=1 || desc->nb_layers!=1 ||
               desc->layers[0].format!=DRM_FORMAT_NV12 || desc->layers[0].nb_planes!=2)
                throw std::runtime_error("MPP requires a single linear NV12 DRM buffer");
            const auto& object=desc->objects[0];
            if(object.format_modifier!=DRM_FORMAT_MOD_LINEAR && object.format_modifier!=DRM_FORMAT_MOD_INVALID)
                throw std::runtime_error("Tiled/compressed DRM buffer is unsupported");
            mapping=std::make_unique<Mapping>(object.fd,object.size);
            for(int i=0;i<2;++i) {
                const auto& plane=desc->layers[0].planes[i];
                int rows=i?h/2:h;
                if(plane.object_index!=0 || plane.offset<0 || plane.pitch<w ||
                   size_t(plane.offset)+size_t(plane.pitch)*rows>object.size)
                    throw std::runtime_error("Invalid DRM plane layout");
                src[i]=static_cast<const uint8_t*>(mapping->data)+plane.offset;stride[i]=plane.pitch;
            }
            src[2]=nullptr;stride[2]=0;pixel=AV_PIX_FMT_NV12;
        }
        scale=sws_getCachedContext(scale,w,h,pixel,w,h,AV_PIX_FMT_BGR24,SWS_BICUBIC,nullptr,nullptr,nullptr);
        if(!scale) throw std::runtime_error("Color conversion initialization failed");
        image.create(h,w,CV_8UC3);
        uint8_t* dst[4]={image.data,nullptr,nullptr,nullptr};int pitch[4]={int(image.step),0,0,0};
        if(sws_scale(scale,src,stride,0,h,dst,pitch)!=h) throw std::runtime_error("Incomplete color conversion");
    }
#endif
public:
    VideoSource(const std::string& path,const std::string& selected,int threads=1,
                const std::string& threading="slice"):backend(selected) {
        if(threads<1 || threads>32 || (threading!="slice" && threading!="frame" && threading!="auto"))
            throw std::runtime_error("Invalid decoder thread configuration");
        if(backend=="opencv") {
            if(!cpu.open(path)) throw std::runtime_error("Video open failed");
            rate=cpu.get(cv::CAP_PROP_FPS);count=cpu.get(cv::CAP_PROP_FRAME_COUNT);
            w=cpu.get(cv::CAP_PROP_FRAME_WIDTH);h=cpu.get(cv::CAP_PROP_FRAME_HEIGHT);return;
        }
        if(backend!="rkmpp" && backend!="ffmpeg") throw std::runtime_error("Unknown decoder");
#ifdef WITH_MPP_SOURCE
        try {
            check(avformat_open_input(&format,path.c_str(),nullptr,nullptr),"Open input");
            check(avformat_find_stream_info(format,nullptr),"Stream info");
            stream=av_find_best_stream(format,AVMEDIA_TYPE_VIDEO,-1,-1,nullptr,0);check(stream,"Video stream");
            auto* st=format->streams[stream];auto id=st->codecpar->codec_id;
            const char* name=id==AV_CODEC_ID_HEVC?(backend=="rkmpp"?"hevc_rkmpp":"hevc"):
                             id==AV_CODEC_ID_H264?(backend=="rkmpp"?"h264_rkmpp":"h264"):nullptr;
            if(!name) throw std::runtime_error("Only H264/HEVC input supported");
            const auto* dec=avcodec_find_decoder_by_name(name);
            if(!dec) throw std::runtime_error("Requested FFmpeg decoder is unavailable");
            codec=avcodec_alloc_context3(dec);if(!codec) throw std::bad_alloc();
            check(avcodec_parameters_to_context(codec,st->codecpar),"Codec parameters");
            codec->thread_count=backend=="ffmpeg"?threads:1;
            if(backend=="ffmpeg") codec->thread_type=threading=="slice"?FF_THREAD_SLICE:
                threading=="frame"?FF_THREAD_FRAME:(FF_THREAD_SLICE|FF_THREAD_FRAME);
            check(avcodec_open2(codec,dec,nullptr),"Open decoder");
            rate=av_q2d(av_guess_frame_rate(format,st,nullptr));
            count=st->nb_frames;w=codec->width;h=codec->height;
            if(count<=0 && st->duration>0) count=std::llround(st->duration*av_q2d(st->time_base)*rate);
            packet=av_packet_alloc();frame=av_frame_alloc();if(!packet || !frame) throw std::bad_alloc();
        } catch(...) {cleanup();throw;}
#else
        throw std::runtime_error("Build with FFmpeg development libraries for this decoder");
#endif
    }
    ~VideoSource() {
#ifdef WITH_MPP_SOURCE
        cleanup();
#endif
    }
    double fps() const{return rate;}
    int frames() const{return count;}
    int width() const{return w;}
    int height() const{return h;}
    bool read(cv::Mat& image) {
        if(backend=="opencv") return cpu.read(image);
#ifdef WITH_MPP_SOURCE
        auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while(std::chrono::steady_clock::now()<deadline) {
            int status=avcodec_receive_frame(codec,frame);
            if(status==0) {convert(image);av_frame_unref(frame);return true;}
            if(status==AVERROR_EOF) return false;
            if(status!=AVERROR(EAGAIN)) check(status,"Receive decoded frame");
            if(!pending && !eof) {
                do {
                    av_packet_unref(packet);status=av_read_frame(format,packet);
                    if(status==AVERROR_EOF) {eof=true;break;}
                    check(status,"Read packet");
                } while(packet->stream_index!=stream);
                pending=!eof;
            }
            if(pending || !flushed) {
                status=avcodec_send_packet(codec,pending?packet:nullptr);
                if(status==0) {if(pending) {pending=false;av_packet_unref(packet);} else flushed=true;}
                else if(status!=AVERROR(EAGAIN)) check(status,"Send packet to decoder");
            }
            if(flushed) std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        throw std::runtime_error("Decode timeout");
#else
        return false;
#endif
    }
};
