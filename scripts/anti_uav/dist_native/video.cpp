// Native video decode -> split-core RKNN + ordered GMC -> ordered Dist.
// No Python interpreter, Python bindings, cached observations or frame skipping.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <openssl/evp.h>
#include <dlfcn.h>
#include <sched.h>
#include <sys/utsname.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include "video_source.hpp"
#include "camera_source.hpp"
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static const auto ENTRY=Clock::now();
double ms(Clock::time_point begin) {return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();}
std::string quote(const std::string& value) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:value) {
        if(c=='"' || c=='\\') out<<'\\'<<c;
        else if(c<32) out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
        else out<<c;
    }
    out<<'"';return out.str();
}
std::string sha256(const std::string& path) {
    std::ifstream in(path,std::ios::binary);if(!in) throw std::runtime_error("Cannot hash "+path);
    std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),EVP_MD_CTX_free);
    if(!ctx || EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)!=1) throw std::runtime_error("SHA256 init failed");
    std::array<char,65536> block;
    while(in) {in.read(block.data(),block.size());if(EVP_DigestUpdate(ctx.get(),block.data(),in.gcount())!=1) throw std::runtime_error("SHA256 update failed");}
    if(!in.eof()) throw std::runtime_error("SHA256 read failed");
    unsigned char digest[EVP_MAX_MD_SIZE];unsigned size=0;
    if(EVP_DigestFinal_ex(ctx.get(),digest,&size)!=1) throw std::runtime_error("SHA256 final failed");
    std::ostringstream out;for(unsigned i=0;i<size;++i) out<<std::hex<<std::setw(2)<<std::setfill('0')<<int(digest[i]);
    return out.str();
}
struct Library {
    void* handle;
    explicit Library(const std::string& path):handle(dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL)) {
        if(!handle) throw std::runtime_error(dlerror());
    }
    ~Library(){dlclose(handle);}
    template<typename T> T get(const char* name) {
        auto p=dlsym(handle,name);if(!p) throw std::runtime_error(std::string("Missing symbol ")+name);
        return reinterpret_cast<T>(p);
    }
};
void affinity(const std::vector<int>& cpus) {
    cpu_set_t mask;CPU_ZERO(&mask);
    for(int cpu:cpus) {if(cpu<0 || cpu>=CPU_SETSIZE) throw std::runtime_error("Invalid CPU");CPU_SET(cpu,&mask);}
    if(sched_setaffinity(0,sizeof(mask),&mask)) throw std::runtime_error("CPU affinity failed");
}
std::string hardware() {
    std::map<std::string,std::string> fields;
    utsname info{};uname(&info);fields["host"]=info.nodename;fields["kernel"]=info.release;
    auto add=[&](const fs::path& path) {std::ifstream in(path);std::string value;if(std::getline(in,value)) fields[path.string()]=value;};
    for(const auto& base:{"/sys/class/devfreq","/sys/class/thermal","/sys/devices/system/cpu/cpufreq"}) {
        std::error_code ec;
        for(fs::directory_iterator it(base,ec),end;!ec && it!=end;it.increment(ec)) {
            auto path=it->path();std::string name=path.filename();
            if(std::string(base).find("devfreq")!=std::string::npos) {add(path/"cur_freq");add(path/"governor");}
            else if(name.rfind("thermal_zone",0)==0) add(path/"temp");
            else if(name.rfind("policy",0)==0) add(path/"scaling_cur_freq");
        }
    }
    std::ostringstream out;out<<'{';bool first=true;
    for(auto& item:fields) {if(!first) out<<',';first=false;out<<quote(item.first)<<':'<<quote(item.second);}
    out<<'}';return out.str();
}
std::string stats(std::vector<double> values) {
    if(values.empty()) return "null";
    double mean=std::accumulate(values.begin(),values.end(),0.)/values.size();
    std::sort(values.begin(),values.end());
    auto q=[&](double fraction) {double i=(values.size()-1)*fraction;size_t l=size_t(i),u=std::min(l+1,values.size()-1);return values[l]+(values[u]-values[l])*(i-l);};
    std::ostringstream out;out<<std::setprecision(17)<<"{\"mean\":"<<mean<<",\"p50\":"<<q(.5)<<",\"p95\":"<<q(.95)<<",\"max\":"<<values.back()<<'}';
    return out.str();
}
struct Args {
    std::string model,detector,gmc,tracker,video,output,cpus="4,5,6,7",decoder="opencv",preprocess="opencv";
    int workers=3,inflight=9,frames=0,warmup=100,decode_threads=1;
    std::string decode_threading="slice";
    std::string camera_policy="latest",camera_format;
    int camera_buffers=4,npu_warmup=0;
    double camera_fps=120;
    double conf=.03,iou=.45;
    bool save=false,detector_only=false,pyramid_cache=true;
};
Args parse(int argc,char** argv) {
    Args a;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--save-observations") {a.save=true;continue;}
        if(key=="--detector-only") {a.detector_only=true;continue;}
        if(key=="--no-pyramid-cache") {a.pyramid_cache=false;continue;}
        if(i+1==argc) throw std::runtime_error("Missing value: "+key);
        std::string value=argv[++i];
        if(key=="--model") a.model=value;
        else if(key=="--detector-library") a.detector=value;
        else if(key=="--tracker-library") a.tracker=value;
        else if(key=="--gmc-library") a.gmc=value;
        else if(key=="--video") a.video=value;
        else if(key=="--output") a.output=value;
        else if(key=="--cpus") a.cpus=value;
        else if(key=="--decoder") a.decoder=value;
        else if(key=="--decode-threads") a.decode_threads=std::stoi(value);
        else if(key=="--decode-threading") a.decode_threading=value;
        else if(key=="--preprocess") a.preprocess=value;
        else if(key=="--camera-format") a.camera_format=value;
        else if(key=="--camera-policy") a.camera_policy=value;
        else if(key=="--camera-buffers") a.camera_buffers=std::stoi(value);
        else if(key=="--camera-fps") a.camera_fps=std::stod(value);
        else if(key=="--npu-warmup") a.npu_warmup=std::stoi(value);
        else if(key=="--workers") a.workers=std::stoi(value);
        else if(key=="--inflight") a.inflight=std::stoi(value);
        else if(key=="--frames") a.frames=std::stoi(value);
        else if(key=="--warmup") a.warmup=std::stoi(value);
        else if(key=="--conf") a.conf=std::stod(value);
        else if(key=="--iou") a.iou=std::stod(value);
        else throw std::runtime_error("Unknown option: "+key);
    }
    if(a.model.empty() || a.detector.empty() || a.video.empty() || a.output.empty() ||
       (!a.detector_only && (a.tracker.empty() || a.gmc.empty())) || a.workers<1 || a.workers>3 ||
       a.inflight<a.workers || a.warmup<0 || a.frames<0 ||
       !std::isfinite(a.conf) || a.conf<0 || a.conf>1 ||
       !std::isfinite(a.iou) || a.iou<0 || a.iou>1) throw std::runtime_error("Invalid arguments");
    if((a.decoder!="opencv" && a.decoder!="rkmpp" && a.decoder!="ffmpeg" && a.decoder!="v4l2") ||
       (a.preprocess!="opencv" && a.preprocess!="rga" && a.preprocess!="fused")) throw std::runtime_error("Invalid image backend");
    if(a.npu_warmup<0 || a.npu_warmup>100 || (a.decoder=="v4l2" &&
       (a.camera_format!="raw8-gray" || a.frames<=a.warmup || !std::isfinite(a.camera_fps) ||
        a.camera_fps<=0 || (a.camera_policy!="latest" && a.camera_policy!="fifo"))))
        throw std::runtime_error("Camera requires raw8-gray, explicit frame count and valid rate/policy");
    return a;
}
struct Job {
    int index,n=0,bad=0;
    cv::Mat frame;
    Clock::time_point begin;
    double decode=0,gmc=0;
    double ready_ms=0,dispatch_ms=0,worker_start_ms=0,detected_ms=0;
    CaptureStamp capture;
    std::array<double,3> native{};
    std::array<double,6> warp{1,0,0,0,1,0};
    std::array<float,500> boxes{};
    bool detected=false,motion=false;
};
using JobPtr=std::shared_ptr<Job>;
struct Work {
    std::mutex mutex;
    std::condition_variable cv;
    bool stop=false,done=false;
    int inflight=0;
    std::exception_ptr failure;
    std::deque<int> available;
    std::deque<JobPtr> ordered,motion;
    std::array<JobPtr,3> pending;
    void fail() {std::lock_guard<std::mutex> lock(mutex);if(!failure) failure=std::current_exception();stop=true;cv.notify_all();}
};
struct Threads {
    Work& work;
    std::vector<std::thread> threads;
    ~Threads() {
        {std::lock_guard<std::mutex> lock(work.mutex);work.stop=true;work.cv.notify_all();}
        for(auto& thread:threads) if(thread.joinable()) thread.join();
    }
};
int run(const Args& args) {
    std::vector<int> cpus;std::stringstream cpu_text(args.cpus);std::string part;
    while(std::getline(cpu_text,part,',')) cpus.push_back(std::stoi(part));
    if(cpus.empty()) throw std::runtime_error("Empty CPU set");
    affinity(cpus);cv::setNumThreads(1);
    if(fs::exists(args.output)) throw std::runtime_error("Output already exists");
    fs::create_directories(args.output);
    const auto before=hardware();
    std::unique_ptr<VideoSource> cap;
    std::unique_ptr<CameraSource> camera;
    if(args.decoder=="v4l2") camera=std::make_unique<CameraSource>(args.video,args.camera_policy=="latest",args.camera_buffers);
    else cap=std::make_unique<VideoSource>(args.video,args.decoder,args.decode_threads,args.decode_threading);
    double fps=camera?args.camera_fps:cap->fps();
    int total=args.frames?args.frames:cap->frames();
    if(fps<=0 || total<=args.warmup) throw std::runtime_error("Invalid video FPS/count/warmup");
    int source_w=camera?camera->width():cap->width(),source_h=camera?camera->height():cap->height();
    Library det(args.detector);
    if(args.preprocess=="rga" && !det.get<int(*)()>("au_detector_rga_supported")())
        throw std::runtime_error("RGA detector library required");
    if(args.preprocess=="fused" && !det.get<int(*)()>("au_detector_fused_supported")())
        throw std::runtime_error("Fused preprocessing detector library required");
    auto create=det.get<int(*)(const char*,const char**,int,int,void**)>("au_detector_create_pool");
    auto infer=det.get<int(*)(void*,unsigned char*,int,int,size_t,float,float,int,float*,double*,int,int)>("au_detector_infer_image");
    auto destroy=det.get<void(*)(void*)>("au_detector_destroy");
    auto det_error=det.get<const char*(*)()>("au_detector_error");
    const char* masks[]={"0","1","2"};void* handles[3]{};
    auto loading=Clock::now();
    if(create(args.model.c_str(),masks,args.workers,1,handles)) throw std::runtime_error(det_error());
    std::vector<std::unique_ptr<void,decltype(destroy)>> owned;
    for(int i=0;i<args.workers;++i) owned.emplace_back(handles[i],destroy);
    double load_ms=ms(loading);
    auto warmup_start=Clock::now();
    if(args.npu_warmup) {
        cv::Mat blank(source_h,source_w,camera?CV_8UC1:CV_8UC3,cv::Scalar::all(114));
        std::array<float,500> boxes{};std::array<double,3> times{};
        for(int n=0;n<args.npu_warmup;++n) for(int w=0;w<args.workers;++w)
            if(infer(handles[w],blank.data,source_w,source_h,blank.step,args.conf,args.iou,100,
                     boxes.data(),times.data(),args.preprocess=="fused"?3:1,blank.channels())<0)
                throw std::runtime_error(det_error());
    }
    double npu_warmup_ms=ms(warmup_start);
    std::unique_ptr<Library> track_lib,flow_lib;
    using Deleter=void(*)(void*);
    std::unique_ptr<void,Deleter> tracker(nullptr,+[](void*){}),flow(nullptr,+[](void*){});
    int(*track_update)(void*,const float*,int,const double*,int*,int)=nullptr;
    int(*flow_apply)(void*,unsigned char*,int,int,size_t,int,double*)=nullptr;
    void(*flow_stats)(void*,uint64_t*,double*)=nullptr;
    const char*(*track_error)()=nullptr;
    const char*(*flow_error)()=nullptr;
    if(!args.detector_only) {
        track_lib=std::make_unique<Library>(args.tracker);flow_lib=std::make_unique<Library>(args.gmc);
        auto create_track=track_lib->get<void*(*)(double,double,double,double,double,int)>("dist_create");
        tracker={create_track(fps,.03,.01,.10,.8,30),track_lib->get<Deleter>("dist_destroy")};
        track_error=track_lib->get<const char*(*)()>("dist_error");
        if(!tracker) throw std::runtime_error(track_error());
        track_update=track_lib->get<decltype(track_update)>("dist_update");
        auto create_flow=flow_lib->get<void*(*)(int,int,int,int,int)>("gmc_create");
        flow={create_flow(320,128,5,args.pyramid_cache,1),flow_lib->get<Deleter>("gmc_destroy")};
        flow_error=flow_lib->get<const char*(*)()>("gmc_error");
        if(!flow) throw std::runtime_error(flow_error());
        flow_apply=flow_lib->get<decltype(flow_apply)>("gmc_apply");flow_stats=flow_lib->get<decltype(flow_stats)>("gmc_stats");
    }
    auto model_sha=sha256(args.model);
    std::ofstream observations;
    if(args.save) {observations.open(fs::path(args.output)/"observations.jsonl");observations.exceptions(std::ios::badbit|std::ios::failbit);observations<<std::setprecision(17);}
    std::ofstream latencies;
    if(camera) {
        latencies.open(fs::path(args.output)/"latency.csv");latencies.exceptions(std::ios::badbit|std::ios::failbit);
        latencies<<"index,sequence,flags,timestamp_valid,frame_ms,dequeue_ms,output_ms,read_ms,copy_ms,dispatch_wait_ms,worker_wait_ms,preprocess_ms,npu_ms,postprocess_ms,gmc_ms,association_ms,read_to_output_ms,frame_to_output_ms\n"<<std::setprecision(17);
    }
    Work work;Threads threads{work,{}};
    std::array<int,3> assigned{};
    for(int worker=0;worker<args.workers;++worker) {
        work.available.push_back(worker);
        threads.threads.emplace_back([&,worker] {
            try {
                affinity({cpus[worker%cpus.size()]});
                while(true) {
                    JobPtr job;
                    {std::unique_lock<std::mutex> lock(work.mutex);work.cv.wait(lock,[&]{return work.stop || work.pending[worker] || work.done;});
                     if(work.stop || !work.pending[worker]) return;job=std::move(work.pending[worker]);}
                    int mode=args.preprocess=="fused"?3:args.preprocess=="rga"?2:1;
                    job->worker_start_ms=monotonic_ms();
                    int count=infer(handles[worker],job->frame.data,job->frame.cols,job->frame.rows,job->frame.step,args.conf,args.iou,100,job->boxes.data(),job->native.data(),mode,job->frame.channels());
                    if(count<0) throw std::runtime_error(det_error());
                    for(int i=0;i<count;++i) {
                        float* b=job->boxes.data()+5*i;
                        if(b[2]>b[0] && b[3]>b[1]) {std::copy(b,b+5,job->boxes.data()+5*job->n);++job->n;}
                        else ++job->bad;
                    }
                    job->detected_ms=monotonic_ms();
                    {std::lock_guard<std::mutex> lock(work.mutex);job->detected=true;work.available.push_back(worker);work.cv.notify_all();}
                }
            } catch(...) {work.fail();}
        });
    }
    if(flow) threads.threads.emplace_back([&] {
        try {
            affinity(cpus);cv::setRNGSeed(20260917);
            while(true) {
                JobPtr job;
                {std::unique_lock<std::mutex> lock(work.mutex);work.cv.wait(lock,[&]{return work.stop || !work.motion.empty() || work.done;});
                 if(work.stop || work.motion.empty()) return;job=work.motion.front();work.motion.pop_front();}
                auto start=Clock::now();
                if(flow_apply(flow.get(),job->frame.data,job->frame.cols,job->frame.rows,job->frame.step,job->frame.channels(),job->warp.data())) throw std::runtime_error(flow_error());
                job->gmc=ms(start);
                {std::lock_guard<std::mutex> lock(work.mutex);job->motion=true;work.cv.notify_all();}
            }
        } catch(...) {work.fail();}
    });
    auto started=Clock::now(),measured=started;
    threads.threads.emplace_back([&] {
        try {
            for(int index=0;index<total;++index) {
                {std::unique_lock<std::mutex> lock(work.mutex);work.cv.wait(lock,[&]{return work.stop || (work.inflight<args.inflight && (!camera || !work.available.empty()));});
                 if(work.stop) return;++work.inflight;}
                auto job=std::make_shared<Job>();job->index=index;job->begin=Clock::now();
                if(camera) camera->read(job->frame,job->capture);
                else if(!cap->read(job->frame)) throw std::runtime_error("Decode failed at "+std::to_string(index));
                job->decode=ms(job->begin);job->ready_ms=monotonic_ms();job->motion=!flow;
                {std::unique_lock<std::mutex> lock(work.mutex);work.cv.wait(lock,[&]{return work.stop || !work.available.empty();});
                 if(work.stop) return;
                 int worker=work.available.front();work.available.pop_front();++assigned[worker];
                 job->dispatch_ms=monotonic_ms();
                 work.pending[worker]=job;work.ordered.push_back(job);if(flow) work.motion.push_back(job);work.cv.notify_all();}
            }
            {std::lock_guard<std::mutex> lock(work.mutex);work.done=true;work.cv.notify_all();}
        } catch(...) {work.fail();}
    });
    std::array<std::vector<double>,13> timing;
    uint64_t sequence_gaps=0;uint32_t last_sequence=0;bool have_sequence=false;
    cv::Mat first_image;
    std::vector<std::string> samples;
    std::string first_result;
    int detections=0,tracks=0,bad=0;
    for(int index=0;index<total;++index) {
        JobPtr job;
        {std::unique_lock<std::mutex> lock(work.mutex);work.cv.wait(lock,[&]{return work.failure || !work.ordered.empty() || work.done;});
         if(work.failure) std::rethrow_exception(work.failure);
         if(work.ordered.empty()) throw std::runtime_error("Incomplete video");
         job=work.ordered.front();work.ordered.pop_front();
         work.cv.wait(lock,[&]{return work.failure || (job->detected && job->motion);});
         if(work.failure) std::rethrow_exception(work.failure);}
        if(job->index!=index) throw std::runtime_error("Out-of-order result");
        auto association_start=Clock::now();std::array<int,200> pairs{};int ntracks=0;
        if(tracker) {
            ntracks=track_update(tracker.get(),job->boxes.data(),job->n,job->warp.data(),pairs.data(),100);
            if(ntracks<0) throw std::runtime_error(track_error());
        }
        double association_ms=ms(association_start),latency=ms(job->begin);
        double output_ms=monotonic_ms();
        if(camera) {
            if(have_sequence) sequence_gaps+=uint32_t(job->capture.sequence-last_sequence)-1;
            have_sequence=true;last_sequence=job->capture.sequence;
            auto& c=job->capture;
            latencies<<index<<','<<c.sequence<<','<<c.flags<<','<<c.valid<<','<<c.frame_ms<<','<<c.dequeue_ms<<','<<output_ms<<','<<job->decode<<','<<c.copy_ms<<','<<job->dispatch_ms-job->ready_ms<<','<<job->worker_start_ms-job->dispatch_ms<<','<<job->native[0]<<','<<job->native[1]<<','<<job->native[2]<<','<<job->gmc<<','<<association_ms<<','<<latency<<',';
            if(c.valid) latencies<<output_ms-c.frame_ms;latencies<<'\n';
        }
        if(index==0) {
            if(camera) first_image=job->frame;
            std::ostringstream out;out<<std::setprecision(17)<<"{\"process_entry_ms\":"<<ms(ENTRY)<<",\"frame_read_ms\":"<<latency<<",\"decode_ms\":"<<job->decode<<",\"gmc_ms\":"<<job->gmc<<",\"association_ms\":"<<association_ms;
            if(camera) {
                out<<",\"streamon_to_output_ms\":"<<output_ms-camera->stream_start_ms<<",\"driver_to_output_ms\":";
                if(job->capture.valid) out<<output_ms-job->capture.frame_ms;else out<<"null";
            }
            out<<'}';
            first_result=out.str();std::cout<<"{\"event\":\"first_result\","<<first_result.substr(1)<<std::endl;
        }
        if(index>=args.warmup) {
            const double values[]={job->decode,job->native[0],job->native[1],job->native[2],job->gmc,association_ms,latency};
            for(int j=0;j<7;++j) timing[j].push_back(values[j]);
            timing[7].push_back(job->dispatch_ms-job->ready_ms);
            timing[8].push_back(job->worker_start_ms-job->dispatch_ms);
            timing[9].push_back(output_ms-job->detected_ms);
            if(camera) {
                timing[10].push_back(job->capture.copy_ms);
                if(job->capture.valid) {
                    timing[11].push_back(job->capture.dequeue_ms-job->capture.frame_ms);
                    timing[12].push_back(output_ms-job->capture.frame_ms);
                }
            }
        }
        if(args.save) {
            auto& out=observations;out<<"{\"frame_index\":"<<index<<",\"boxes_xyxy_score\":[";
            for(int i=0;i<job->n;++i) {if(i) out<<',';out<<'[';for(int j=0;j<5;++j) {if(j) out<<',';out<<job->boxes[5*i+j];}out<<']';}
            out<<"],\"displayed_tracks\":[";
            for(int i=0;i<ntracks;++i) {
                int det_index=pairs[2*i+1];if(det_index<0 || det_index>=job->n) throw std::runtime_error("Invalid associated detection");
                float* b=job->boxes.data()+5*det_index;
                if(i) out<<',';out<<"{\"id\":"<<pairs[2*i]<<",\"detection_index\":"<<det_index<<",\"score\":"<<b[4]<<",\"box\":[";
                for(int k=0;k<4;++k) {if(k) out<<',';out<<b[k];}out<<"]}";
            }
            out<<"],\"warp\":";
            if(flow) {out<<'[';for(int r=0;r<2;++r) {if(r) out<<',';out<<'[';for(int c=0;c<3;++c) {if(c) out<<',';out<<job->warp[3*r+c];}out<<']';}out<<']';}
            else out<<"null";out<<"}\n";
        }
        detections+=job->n;tracks+=ntracks;bad+=job->bad;
        {std::lock_guard<std::mutex> lock(work.mutex);--work.inflight;work.cv.notify_all();}
        if(index+1==args.warmup) measured=Clock::now();
        if((index+1)%500==0) {
            double elapsed=ms(started)/1000;
            std::ostringstream sample;sample<<std::setprecision(17)<<"{\"frame\":"<<index+1<<",\"elapsed_seconds\":"<<elapsed<<",\"hardware\":"<<hardware()<<'}';samples.push_back(sample.str());
            std::cout<<"{\"event\":\"progress\",\"frame\":"<<index+1<<",\"elapsed\":"<<elapsed<<'}'<<std::endl;
        }
    }
    double seconds=ms(measured)/1000,all_seconds=ms(started)/1000;
    for(auto& thread:threads.threads) thread.join();
    if(camera) {latencies.close();cv::imwrite((fs::path(args.output)/"first_raw_gray.png").string(),first_image);}
    if(args.save) observations.close();
    std::ofstream out(fs::path(args.output)/"summary.json");out.exceptions(std::ios::badbit|std::ios::failbit);out<<std::setprecision(17);
    out<<"{\"runtime\":\"native_cpp_no_python\",\"opencv_version\":"<<quote(CV_VERSION)<<",\"frames\":"<<total<<",\"measured_frames\":"<<total-args.warmup<<",\"measured_seconds\":"<<seconds<<",\"steady_fps\":"<<(total-args.warmup)/seconds<<",\"all_frames_fps\":"<<total/all_seconds;
    out<<",\"model_load_ms\":"<<load_ms<<",\"first_result\":"<<first_result<<",\"model_sha256\":"<<quote(model_sha)<<",\"detector_library_sha256\":"<<quote(sha256(args.detector));
    out<<",\"decoder_backend\":"<<quote(args.decoder)<<",\"preprocess_backend\":"<<quote(args.preprocess);
    out<<",\"npu_warmup_per_worker\":"<<args.npu_warmup<<",\"npu_warmup_ms\":"<<npu_warmup_ms;
    if(camera) out<<",\"camera\":{\"format\":\"raw8-gray\",\"policy\":"<<quote(args.camera_policy)<<",\"buffers\":"<<camera->buffer_count()<<",\"drained_frames\":"<<camera->discarded<<",\"sequence_gaps\":"<<sequence_gaps<<",\"timestamp_scope\":\"Driver MONOTONIC frame timestamp, not verified sensor exposure time; no display\"}";
    if(args.decoder=="ffmpeg") out<<",\"software_decode_threads\":"<<args.decode_threads
        <<",\"software_decode_threading\":"<<quote(args.decode_threading);
    if(tracker) out<<",\"tracker_library_sha256\":"<<quote(sha256(args.tracker))<<",\"gmc_library_sha256\":"<<quote(sha256(args.gmc));
    out<<",\"args\":{\"conf\":"<<args.conf<<",\"iou\":"<<args.iou<<",\"actual_conf_float32\":"<<float(args.conf)<<",\"actual_iou_float32\":"<<float(args.iou)<<",\"workers\":"<<args.workers<<",\"inflight\":"<<args.inflight<<",\"warmup\":"<<args.warmup<<",\"detector_only\":"<<(args.detector_only?"true":"false")<<",\"pyramid_cache\":"<<(args.pyramid_cache?"true":"false")<<",\"cpus\":"<<quote(args.cpus)<<",\"video\":"<<quote(args.video)<<",\"model\":"<<quote(args.model)<<",\"save_observations\":"<<(args.save?"true":"false")<<'}';
    out<<",\"input_wh\":[960,544],\"source_wh\":["<<source_w<<','<<source_h<<"],\"source_fps\":"<<fps<<",\"npu_core_masks\":[";
    for(int i=0;i<args.workers;++i) {if(i) out<<',';out<<quote(masks[i]);}out<<"],\"npu_worker_frame_counts\":[";
    for(int i=0;i<args.workers;++i) {if(i) out<<',';out<<assigned[i];}out<<']';
    out<<",\"detector_count\":"<<detections<<",\"displayed_tracks\":"<<tracks<<",\"rejected_degenerate_boxes\":"<<bad<<",\"stages_ms\":{";
    const char* names[]={"decode","preprocess","npu","postprocess","gmc","association","ordered_latency","dispatch_wait","worker_wait","detector_done_to_output","capture_copy","driver_to_dequeue","driver_to_output"};
    for(int i=0;i<13;++i) {if(i) out<<',';out<<quote(names[i])<<':'<<stats(timing[i]);}out<<'}';
    if(flow) {
        uint64_t counts[4];double seconds[5];flow_stats(flow.get(),counts,seconds);
        out<<",\"compact_gmc_counts\":{";const char* cn[]={"frames","estimated","identity_fallback","refreshes"};
        for(int i=0;i<4;++i) {if(i) out<<',';out<<quote(cn[i])<<':'<<counts[i];}out<<"},\"compact_gmc_seconds\":{";
        const char* sn[]={"preprocess","pyramid","optical_flow","ransac","features"};for(int i=0;i<5;++i) {if(i) out<<',';out<<quote(sn[i])<<':'<<seconds[i];}out<<'}';
    }
    out<<",\"hardware_before\":"<<before<<",\"hardware_after\":"<<hardware()<<",\"hardware_samples\":[";
    for(size_t i=0;i<samples.size();++i) {if(i) out<<',';out<<samples[i];}
    out<<"],\"scope\":"<<quote(camera?"Native live raw8 camera + RKNN + GMC + Dist. Dropped frames explicitly counted. No display/encoding. Tracker thresholds unchanged; live tracking accuracy not validated.":"Native C++ video pipeline; no Python, cached detections, frame skipping, rendering or encoding.")<<"}\n";out.close();
    std::cout<<"{\"event\":\"complete\",\"frames\":"<<total<<",\"fps\":"<<(total-args.warmup)/seconds<<'}'<<std::endl;
    return 0;
}
int main(int argc,char** argv) {
    try {return run(parse(argc,argv));}
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
