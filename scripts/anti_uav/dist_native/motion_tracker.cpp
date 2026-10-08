// SPDX-License-Identifier: AGPL-3.0-only
// A new motion-aware algorithm, NOT an equivalent port of public Dist/OC-SORT.
// Reuse the tested matrix/LAP primitives and retain the original Dist ABI unchanged.
#include "tracker.cpp"
#include "motion_tracker.hpp"
#include <deque>
#include <limits>
#include <numeric>
#include <cstdint>

namespace motion {
struct Observation { double time; std::array<double,4> box; };
struct Config {
    std::array<double,14> v;
    explicit Config(const double* data,int n) {
        if(!data || n!=14) throw std::runtime_error("Expected 14 motion configuration values");
        std::copy(data,data+n,v.begin());
        for(double x:v) if(!std::isfinite(x)) throw std::runtime_error("Non-finite motion configuration");
        if(!(0<=v[1] && v[1]<v[0] && v[0]<=v[2] && v[2]<=1 && v[3]>0 && v[4]>0 &&
             v[5]>0 && v[6]>0 && v[7]>=0 && v[8]>0 && v[9]>0 && v[10]>0 &&
             v[11]>=1 && v[11]<=v[12] && v[12]<=32 && v[11]==int(v[11]) &&
             v[12]==int(v[12]) && v[13]>=0 && v[13]<.5))
            throw std::runtime_error("Invalid motion configuration");
    }
};
struct Target {
    int id,index,frame,age=1;
    bool confirmed=false;
    double last_time,accel=0,camera_sigma=0;
    V8 mean;
    M8 covariance;
    std::deque<Observation> history;
    std::deque<unsigned char> hits{1};
    Target(const Detection& d,int fid,int tid,double time,const Config& c):
        id(tid),index(d.index),frame(fid),last_time(time) {
        for(int i=0;i<4;++i) mean(i,0)=d.measurement[i];
        for(int i=0;i<8;++i) {
            double sigma=i<2?std::max(c.v[5],.03*d.measurement[2+i]):
                i<4?std::max(1.,.05*d.measurement[i]):i<6?1200.:20.;
            covariance(i,i)=sigma*sigma;
        }
        history.push_back({time,{mean(0,0),mean(1,0),mean(2,0),mean(3,0)}});
        confirmed=c.v[11]==1;
    }
    std::array<float,4> bounds() const {
        double w=std::max(.01,mean(2,0)),h=std::max(.01,mean(3,0));
        return {float(mean(0,0)-w/2),float(mean(1,0)-h/2),
                float(mean(0,0)+w/2),float(mean(1,0)+h/2)};
    }
    void predict(double dt,const double* h,double quality,const Config& c) {
        // Warp observations as well as the state, so measured velocity is camera compensated.
        M8 j;
        for(int i=0;i<8;i+=4) {
            j(i,i)=h[0];j(i,i+1)=h[1];j(i+1,i)=h[3];j(i+1,i+1)=h[4];
            j(i+2,i+2)=std::abs(h[0]);j(i+2,i+3)=std::abs(h[1]);
            j(i+3,i+2)=std::abs(h[3]);j(i+3,i+3)=std::abs(h[4]);
        }
        mean=mul(j,mean);mean(0,0)+=h[2];mean(1,0)+=h[5];
        covariance=mul(mul(j,covariance),transpose(j));
        for(auto& obs:history) {
            double x=obs.box[0],y=obs.box[1],w=obs.box[2],height=obs.box[3];
            obs.box={h[0]*x+h[1]*y+h[2],h[3]*x+h[4]*y+h[5],
                     std::abs(h[0])*w+std::abs(h[1])*height,
                     std::abs(h[3])*w+std::abs(h[4])*height};
        }
        M8 f;for(int i=0;i<8;++i) f(i,i)=1;
        for(int i=0;i<4;++i) f(i,i+4)=dt;
        mean=mul(f,mean);covariance=mul(mul(f,covariance),transpose(f));
        double sigma_a=std::max(c.v[6],accel*.5);
        for(int axis=0;axis<4;++axis) {
            double a=axis<2?sigma_a:30.,q=a*a;
            covariance(axis,axis)+=q*dt*dt*dt*dt/4;
            covariance(axis,axis+4)+=q*dt*dt*dt/2;
            covariance(axis+4,axis)+=q*dt*dt*dt/2;
            covariance(axis+4,axis+4)+=q*dt*dt;
        }
        camera_sigma=(1-quality)*c.v[7]*dt;
        for(int i=0;i<2;++i) covariance(i,i)+=camera_sigma*camera_sigma+c.v[5]*c.v[5]*dt*c.v[4];
        mean(2,0)=std::max(.01,mean(2,0));mean(3,0)=std::max(.01,mean(3,0));
        hits.push_back(0);if(hits.size()>size_t(c.v[12])) hits.pop_front();++age;
    }
    double cost(const Detection& d,double time,int fid,const Config& c) const {
        double dx=d.measurement[0]-mean(0,0),dy=d.measurement[1]-mean(1,0);
        double sx=std::max(c.v[5],.03*d.measurement[2]),sy=std::max(c.v[5],.03*d.measurement[3]);
        double a=covariance(0,0)+sx*sx,b=covariance(0,1),e=covariance(1,1)+sy*sy;
        double determinant=a*e-b*b;
        if(!(determinant>0)) return 100.;
        double nis=(e*dx*dx-2*b*dx*dy+a*dy*dy)/determinant;
        double gap=std::max(1./c.v[4],time-last_time);
        double radius=std::min(c.v[9],std::max(12.,c.v[8]*gap+3*std::hypot(sx,sy)));
        // Score both smooth motion and a bounded maneuver hypothesis in the PRIMARY cost.
        double maneuver_sigma=radius*.25;
        double am=a+maneuver_sigma*maneuver_sigma,em=e+maneuver_sigma*maneuver_sigma;
        double maneuver_nis=(em*dx*dx-2*b*dx*dy+am*dy*dy)/(am*em-b*b);
        if((nis>c.v[10] && maneuver_nis>c.v[10]) || std::hypot(dx,dy)>radius) return 100.;
        // A large covariance is not evidence of a good identity match: penalize its volume.
        double reference_variance=std::pow(std::max(c.v[5]*2,.1*std::sqrt(d.measurement[2]*d.measurement[3])),2)
                                  +camera_sigma*camera_sigma;
        double likelihood=.9*std::exp(-nis*.5)*reference_variance/std::sqrt(determinant)+
            .1*std::exp(-maneuver_nis*.5)*(reference_variance+maneuver_sigma*maneuver_sigma)/std::sqrt(am*em-b*b);
        likelihood=std::min(1.,likelihood);
        double motion_cost=std::min(1.,-2*std::log(std::max(1e-12,likelihood))/c.v[10]);
        double shape=std::abs(std::log(d.measurement[2]/std::max(.01,mean(2,0))))+
                     std::abs(std::log(d.measurement[3]/std::max(.01,mean(3,0))));
        if(shape>std::log(16.)) return 100.;
        double direction=0;
        if(history.size()>=2) {
            double vx=mean(4,0),vy=mean(5,0);
            double mx=d.measurement[0]-history.back().box[0],my=d.measurement[1]-history.back().box[1];
            double speed=std::hypot(vx,vy),movement=std::hypot(mx,my);
            if(speed>30 && movement>c.v[5]*2)
                direction=(1-std::clamp((vx*mx+vy*my)/(speed*movement),-1.,1.))*.5;
        }
        double gap_penalty=.12*std::min(1.,(time-last_time)/c.v[3]);
        double continuity_prior=!confirmed?.08:frame==fid-1?0:.12;
        // IoU is reliable for large boxes; tiny boxes need motion-distance evidence instead.
        double size_blend=std::clamp((std::sqrt(d.measurement[2]*d.measurement[3])-16.)/48.,0.,1.);
        return (.55-.30*size_blend)*motion_cost+(.15+.35*size_blend)*distance(bounds(),d.bounds)+
               .10*std::min(1.,shape/std::log(4.))+(.10-.05*size_blend)*direction+.05*(1-d.score)+
               gap_penalty+continuity_prior;
    }
    void observe(const Detection& d,int fid,double time,const Config& c) {
        Matrix<4,4> s,l;
        std::array<double,4> variance{};
        for(int i=0;i<4;++i) {
            double sigma=std::max(i<2?c.v[5]:1.,.03*d.measurement[2+i%2]);variance[i]=sigma*sigma;
            for(int k=0;k<4;++k) s(i,k)=covariance(i,k);
            s(i,i)+=variance[i];
        }
        for(int i=0;i<4;++i) for(int k=0;k<=i;++k) {
            double value=s(i,k);for(int z=0;z<k;++z) value-=l(i,z)*l(k,z);
            if(i==k) {if(!(value>0)) throw std::runtime_error("Invalid motion covariance");l(i,k)=std::sqrt(value);}
            else l(i,k)=value/l(k,k);
        }
        Matrix<8,4> gain;
        for(int row=0;row<8;++row) {
            double y[4]{},x[4]{};
            for(int i=0;i<4;++i) {
                double value=covariance(row,i);for(int k=0;k<i;++k) value-=l(i,k)*y[k];y[i]=value/l(i,i);
            }
            for(int i=3;i>=0;--i) {
                double value=y[i];for(int k=i+1;k<4;++k) value-=l(k,i)*x[k];x[i]=value/l(i,i);gain(row,i)=x[i];
            }
        }
        Matrix<4,1> innovation;
        for(int i=0;i<4;++i) innovation(i,0)=d.measurement[i]-mean(i,0);
        auto correction=mul(gain,innovation);
        M8 residual;for(int i=0;i<8;++i) residual(i,i)=1;
        for(int i=0;i<8;++i) for(int k=0;k<4;++k) residual(i,k)-=gain(i,k);
        auto next=mul(mul(residual,covariance),transpose(residual));
        for(int i=0;i<8;++i) {
            mean(i,0)+=correction(i,0);
            for(int k=0;k<8;++k) for(int z=0;z<4;++z) next(i,k)+=gain(i,z)*variance[z]*gain(k,z);
        }
        covariance=next;
        double dt=time-last_time;
        if(dt>0 && !history.empty()) {
            double vx=(d.measurement[0]-history.back().box[0])/dt;
            double vy=(d.measurement[1]-history.back().box[1])/dt;
            double observed_a=std::min(60000.,std::hypot(vx-mean(4,0),vy-mean(5,0))/dt);
            accel=.8*accel+.2*observed_a;
        }
        history.push_back({time,{d.measurement[0],d.measurement[1],d.measurement[2],d.measurement[3]}});
        while(history.size()>3) history.pop_front();
        // Fit velocity to real observations, not accumulated missing-frame predictions.
        if(history.size()>=2) {
            double mt=0,mx=0,my=0,den=0,nx=0,ny=0;
            for(auto& o:history) {mt+=o.time;mx+=o.box[0];my+=o.box[1];}
            mt/=history.size();mx/=history.size();my/=history.size();
            for(auto& o:history) {double t=o.time-mt;den+=t*t;nx+=t*(o.box[0]-mx);ny+=t*(o.box[1]-my);}
            if(den>1e-12) {
                const double slopes[]={nx/den,ny/den};
                for(int axis=0;axis<2;++axis) {
                    double observed_variance=variance[axis]/den;
                    double prior_variance=covariance(4+axis,4+axis);
                    double weight=std::min(.75,prior_variance/(prior_variance+observed_variance));
                    mean(4+axis,0)=(1-weight)*mean(4+axis,0)+weight*slopes[axis];
                    covariance(4+axis,4+axis)=std::max(prior_variance,weight*weight*observed_variance);
                }
            }
        }
        hits.back()=1;
        if(std::accumulate(hits.begin(),hits.end(),0)>=int(c.v[11])) confirmed=true;
        frame=fid;index=d.index;last_time=time;
    }
};
struct Tracker {
    Config config;
    std::vector<Target> targets;
    double previous_time=-1;
    int frame=0,next_id=1;
    std::array<uint64_t,8> counts{};
    explicit Tracker(const double* data,int n):config(data,n) {}
    void associate(const std::vector<Detection>& dets,double time,std::vector<bool>& used_targets,
                   std::vector<bool>& used_dets,bool low) {
        constexpr double limit=.70;
        std::vector<int> rows,cols;
        for(size_t j=0;j<dets.size();++j)
            if(!used_dets[j] && (low?dets[j].score<config.v[0]:dets[j].score>=config.v[0])) cols.push_back(j);
        for(size_t i=0;i<targets.size();++i) if(!used_targets[i] && (!low || targets[i].confirmed)) rows.push_back(i);
        if(rows.empty() || cols.empty()) return;
        std::vector<std::vector<double>> cost(rows.size(),std::vector<double>(cols.size(),100.));
        for(size_t i=0;i<rows.size();++i) for(size_t j=0;j<cols.size();++j)
            cost[i][j]=targets[rows[i]].cost(dets[cols[j]],time,frame,config);
        // Ambiguous observations remain unassigned; do not manufacture stable identities.
        for(size_t j=0;j<cols.size();++j) {
            double first=100,second=100;
            for(size_t i=0;i<rows.size();++i) {double c=cost[i][j];if(c<first) {second=first;first=c;} else second=std::min(second,c);}
            if(first<limit && second<limit && second-first<config.v[13]) {
                for(auto& row:cost) row[j]=100.;used_dets[cols[j]]=true;++counts[5];
            }
        }
        for(auto& row:cost) {
            auto sorted=row;std::sort(sorted.begin(),sorted.end());
            if(sorted.size()>1 && sorted[1]<limit && sorted[1]-sorted[0]<config.v[13]) {
                for(size_t j=0;j<cols.size();++j) if(row[j]<limit && row[j]-sorted[0]<config.v[13])
                    used_dets[cols[j]]=true;
                std::fill(row.begin(),row.end(),100.);++counts[5];
            }
        }
        for(size_t j=0;j<cols.size();++j) if(used_dets[cols[j]]) for(auto& row:cost) row[j]=100.;
        std::vector<int> relevant;
        for(size_t i=0;i<rows.size();++i) if(*std::min_element(cost[i].begin(),cost[i].end())<limit) relevant.push_back(i);
        if(relevant.empty()) return;
        int nr=relevant.size(),nc=cols.size(),n=nr+nc;
        std::vector<double> padded(n*n,limit/2);
        std::vector<double*> pointers(n);std::vector<int> x(n),y(n);
        for(int i=0;i<n;++i) pointers[i]=padded.data()+i*n;
        for(int i=nr;i<n;++i) for(int j=nc;j<n;++j) pointers[i][j]=0;
        for(int i=0;i<nr;++i) for(int j=0;j<nc;++j) pointers[i][j]=cost[relevant[i]][j];
        if(lapjv_internal(n,pointers.data(),x.data(),y.data())) throw std::runtime_error("Motion LAPJV failed");
        for(int i=0;i<nr;++i) if(x[i]>=0 && x[i]<nc) {
            int ti=rows[relevant[i]],di=cols[x[i]];
            auto& target=targets[ti];auto& d=dets[di];
            if(distance(target.bounds(),d.bounds)>.99999f) ++counts[2];
            bool confirmed=target.confirmed;target.observe(d,frame,time,config);
            ++counts[1];if(!confirmed && target.confirmed) ++counts[4];
            used_targets[ti]=true;used_dets[di]=true;
        }
    }
    void update(const float* boxes,int n,const double* warp,double quality,double time) {
        if(!std::isfinite(time) || time<0 || (previous_time>=0 && time<=previous_time))
            throw std::runtime_error("Motion timestamps must increase strictly");
        if(!std::isfinite(quality) || quality<0 || quality>1) throw std::runtime_error("Invalid GMC quality");
        for(int i=0;i<6;++i) if(!std::isfinite(warp[i])) throw std::runtime_error("Invalid motion warp");
        for(int i=0;i<n;++i) {
            const float* b=boxes+5*i;
            for(int k=0;k<5;++k) if(!std::isfinite(b[k])) throw std::runtime_error("Invalid motion detection");
            if(!(b[2]>b[0] && b[3]>b[1] && b[4]>=0 && b[4]<=1)) throw std::runtime_error("Invalid motion box/score");
        }
        ++frame;++counts[0];
        double dt=previous_time<0?1/config.v[4]:time-previous_time;previous_time=time;
        targets.erase(std::remove_if(targets.begin(),targets.end(),[&](const Target& t) {
            bool expired=time-t.last_time>config.v[3] || (!t.confirmed && t.age>int(config.v[12]) &&
                std::accumulate(t.hits.begin(),t.hits.end(),0)==0);
            if(expired) ++counts[6];return expired;
        }),targets.end());
        for(auto& t:targets) t.predict(dt,warp,quality,config);
        std::vector<Detection> dets;
        for(int i=0;i<n;++i) if(boxes[5*i+4]>config.v[1]) dets.emplace_back(boxes+5*i,i);
        std::vector<bool> used_targets(targets.size()),used_dets(dets.size());
        associate(dets,time,used_targets,used_dets,false);
        associate(dets,time,used_targets,used_dets,true);
        for(size_t j=0;j<dets.size();++j) if(!used_dets[j] && dets[j].score>=config.v[2]) {
            targets.emplace_back(dets[j],frame,next_id++,time,config);++counts[3];
        }
        counts[7]=targets.size();
    }
};
thread_local std::string error;
}
extern "C" {
const char* motion_error() { return motion::error.c_str(); }
void* motion_create(const double* config,int n) {
    try {return new motion::Tracker(config,n);} catch(const std::exception& e) {motion::error=e.what();return nullptr;}
}
void motion_destroy(void* p) {delete static_cast<motion::Tracker*>(p);}
int motion_update(void* p,const float* boxes,int n,const double* warp,double quality,double time,int* pairs,int capacity) {
    try {
        if(!p || n<0 || (n && !boxes) || !warp || !pairs || capacity<0) throw std::runtime_error("Invalid motion ABI input");
        auto& t=*static_cast<motion::Tracker*>(p);t.update(boxes,n,warp,quality,time);int count=0;
        for(auto& track:t.targets) if(track.confirmed && track.frame==t.frame) {
            if(count>=capacity) throw std::runtime_error("Motion output capacity exceeded");
            pairs[2*count]=track.id;pairs[2*count+1]=track.index;++count;
        }
        return count;
    } catch(const std::exception& e) {motion::error=e.what();return -1;}
}
void motion_stats(void* p,uint64_t* counts) {
    auto& t=*static_cast<motion::Tracker*>(p);std::copy(t.counts.begin(),t.counts.end(),counts);
}
}
