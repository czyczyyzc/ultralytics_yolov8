// SPDX-License-Identifier: AGPL-3.0-only
// Behavioral port of Dist-Tracker's Ultralytics BOTSORT/ByteTracker (no ReID).
// Upstream: 396c359e1aa8be4fd5e81a02626cb1ee3867cf7c; see README.md.
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "third_party/lap/lapjv.h"

namespace {
template<int R, int C> struct Matrix {
    std::array<double,R*C> values{};
    double& operator()(int r,int c) { return values[r*C+c]; }
    double operator()(int r,int c) const { return values[r*C+c]; }
};
template<int R,int K,int C> Matrix<R,C> mul(const Matrix<R,K>& a,const Matrix<K,C>& b) {
    Matrix<R,C> result;
    for(int r=0;r<R;++r) for(int c=0;c<C;++c)
        for(int k=0;k<K;++k) result(r,c)+=a(r,k)*b(k,c);
    return result;
}
template<int R,int C> Matrix<C,R> transpose(const Matrix<R,C>& a) {
    Matrix<C,R> result;
    for(int r=0;r<R;++r) for(int c=0;c<C;++c) result(c,r)=a(r,c);
    return result;
}
using V8=Matrix<8,1>;
using M8=Matrix<8,8>;
struct Detection {
    std::array<float,4> tlwh{}, measurement{}, bounds{};
    float score;
    int index;
    Detection(const float* b,int i):score(b[4]),index(i) {
        float w=b[2]-b[0], h=b[3]-b[1];
        float cx=(b[0]+b[2])*.5f, cy=(b[1]+b[3])*.5f;
        // Match float32 xywh, float64 index concatenation, then float32 tlwh.
        float x=static_cast<float>(double(cx)-double(w)*.5);
        float y=static_cast<float>(double(cy)-double(h)*.5);
        tlwh={x,y,w,h}; measurement={x+w*.5f,y+h*.5f,w,h};
        bounds={x,y,x+w,y+h};
    }
};
enum State { New=0,Tracked=1,Lost=2,Removed=3 };
struct Track {
    int id,frame,start,index,state=Tracked;
    bool confirmed, mean32=true;
    float score;
    V8 mean;
    M8 covariance;
    Track(const Detection& d,int fid,int tid):id(tid),frame(fid),start(fid),index(d.index),
        confirmed(fid==1),score(d.score) {
        for(int i=0;i<4;++i) mean(i,0)=d.measurement[i];
        for(int i=0;i<8;++i) {
            double stddev=(i<4?.1:.0625)*mean(2+i%2,0);
            covariance(i,i)=stddev*stddev;
        }
    }
    std::array<float,4> bounds() const {
        if(mean32) {
            float w=mean(2,0),h=mean(3,0),x=float(mean(0,0))-w*.5f,y=float(mean(1,0))-h*.5f;
            return {x,y,x+w,y+h};
        }
        double x=mean(0,0)-mean(2,0)*.5,y=mean(1,0)-mean(3,0)*.5;
        return {float(x),float(y),float(x+mean(2,0)),float(y+mean(3,0))};
    }
    void predict() {
        if(state!=Tracked) { mean(6,0)=0; mean(7,0)=0; }
        M8 f;
        for(int i=0;i<8;++i) f(i,i)=1;
        for(int i=0;i<4;++i) f(i,i+4)=1;
        M8 next=mul(mul(f,covariance),transpose(f));
        for(int i=0;i<8;++i) {
            double stddev=(i<4?.05:.00625)*mean(2+i%2,0);
            next(i,i)+=stddev*stddev;
        }
        mean=mul(f,mean); covariance=next; mean32=false;
    }
    void warp(const double* h) {
        M8 r;
        for(int i=0;i<8;i+=2) {
            r(i,i)=h[0];r(i,i+1)=h[1];r(i+1,i)=h[3];r(i+1,i+1)=h[4];
        }
        mean=mul(r,mean);mean(0,0)+=h[2];mean(1,0)+=h[5];
        covariance=mul(mul(r,covariance),transpose(r));mean32=false;
    }
    void update(const Detection& d,int fid) {
        Matrix<4,4> s,l;
        for(int i=0;i<4;++i) for(int j=0;j<4;++j) s(i,j)=covariance(i,j);
        for(int i=0;i<4;++i) {
            double stddev=.05*mean(2+i%2,0);s(i,i)+=stddev*stddev;
        }
        for(int i=0;i<4;++i) for(int j=0;j<=i;++j) {
            double v=s(i,j);
            for(int k=0;k<j;++k) v-=l(i,k)*l(j,k);
            if(i==j) {
                if(!(v>0)) throw std::runtime_error("Non-positive Kalman covariance");
                l(i,j)=std::sqrt(v);
            } else l(i,j)=v/l(j,j);
        }
        Matrix<8,4> gain;
        for(int row=0;row<8;++row) {
            double y[4]{},x[4]{};
            for(int i=0;i<4;++i) {
                double v=covariance(row,i);
                for(int k=0;k<i;++k) v-=l(i,k)*y[k];
                y[i]=v/l(i,i);
            }
            for(int i=3;i>=0;--i) {
                double v=y[i];for(int k=i+1;k<4;++k) v-=l(k,i)*x[k];
                x[i]=v/l(i,i);gain(row,i)=x[i];
            }
        }
        Matrix<4,1> innovation;
        for(int i=0;i<4;++i) innovation(i,0)=double(d.measurement[i])-mean(i,0);
        auto delta=mul(gain,innovation);auto subtract=mul(mul(gain,s),transpose(gain));
        for(int i=0;i<8;++i) {
            mean(i,0)+=delta(i,0);
            for(int j=0;j<8;++j) covariance(i,j)-=subtract(i,j);
        }
        state=Tracked;confirmed=true;frame=fid;score=d.score;index=d.index;mean32=false;
    }
};
using Ptr=std::shared_ptr<Track>;
using List=std::vector<Ptr>;
float distance(const std::array<float,4>& a,const std::array<float,4>& b) {
    float inter=std::max(0.f,std::min(a[2],b[2])-std::max(a[0],b[0]))*
                std::max(0.f,std::min(a[3],b[3])-std::max(a[1],b[1]));
    float area_b=(b[2]-b[0])*(b[3]-b[1]),area_a=(a[2]-a[0])*(a[3]-a[1]);
    return 1.f-inter/((area_b+area_a-inter)+1e-7f);
}
List join(const List& a,const List& b) {
    List result=a;std::unordered_set<int> seen;
    for(auto& t:a) seen.insert(t->id);
    for(auto& t:b) if(seen.insert(t->id).second) result.push_back(t);
    return result;
}
List sub(const List& a,const List& b) {
    List result;std::unordered_set<int> ids;
    for(auto& t:b) ids.insert(t->id);
    for(auto& t:a) if(!ids.count(t->id)) result.push_back(t);
    return result;
}
struct Matches { std::vector<std::pair<int,int>> pairs; std::vector<int> rows,cols; };
struct Tracker {
    int frame=0,next_id=1,max_lost;
    double high,low,new_score,match;
    List tracked,lost,removed;
    std::vector<double> costs;
    std::vector<double*> cost_rows;
    std::vector<int> x,y;
    Tracker(double fps,double hi,double lo,double ns,double ma,int buffer):
        max_lost(int(fps/30.0*buffer)),high(hi),low(lo),new_score(ns),match(ma) {}
    Matches assign(const List& tracks,const std::vector<Detection>& dets,double threshold) {
        Matches result;int nr=tracks.size(),nc=dets.size();
        if(!nr || !nc) {
            for(int i=0;i<nr;++i) result.rows.push_back(i);
            for(int j=0;j<nc;++j) result.cols.push_back(j);
            return result;
        }
        // Preserve lap.lapjv 0.5.12 finite cost_limit augmentation and tie breaking.
        int n=nr+nc;costs.assign(n*n,threshold/2);cost_rows.resize(n);x.resize(n);y.resize(n);
        for(int i=0;i<n;++i) cost_rows[i]=costs.data()+i*n;
        for(int i=nr;i<n;++i) for(int j=nc;j<n;++j) cost_rows[i][j]=0;
        for(int i=0;i<nr;++i) {
            auto bounds=tracks[i]->bounds();
            for(int j=0;j<nc;++j) cost_rows[i][j]=distance(bounds,dets[j].bounds);
        }
        if(lapjv_internal(n,cost_rows.data(),x.data(),y.data())!=0)
            throw std::runtime_error("LAPJV failed");
        for(int i=0;i<nr;++i) {
            if(x[i]>=0 && x[i]<nc) result.pairs.emplace_back(i,x[i]);
            else result.rows.push_back(i);
        }
        for(int j=0;j<nc;++j) if(y[j]<0 || y[j]>=nr) result.cols.push_back(j);
        return result;
    }
    void update(const float* boxes,int n,const double* warp) {
        ++frame;List active,refind,new_lost,new_removed,confirmed,unconfirmed;
        std::vector<Detection> high_dets,low_dets;
        for(int i=0;i<n;++i) {
            Detection d(boxes+5*i,i);
            if(d.score>=float(high)) high_dets.push_back(d);
            else if(d.score>float(low)) low_dets.push_back(d);
        }
        for(auto& t:tracked) (t->confirmed?confirmed:unconfirmed).push_back(t);
        auto pool=join(confirmed,lost);
        for(auto& t:pool) {t->predict();t->warp(warp);}
        for(auto& t:unconfirmed) t->warp(warp);
        auto first=assign(pool,high_dets,match);
        for(auto [i,j]:first.pairs) {
            bool was_tracked=pool[i]->state==Tracked;
            pool[i]->update(high_dets[j],frame);
            (was_tracked?active:refind).push_back(pool[i]);
        }
        List remaining;
        for(int i:first.rows) if(pool[i]->state==Tracked) remaining.push_back(pool[i]);
        auto second=assign(remaining,low_dets,.5);
        for(auto [i,j]:second.pairs) {remaining[i]->update(low_dets[j],frame);active.push_back(remaining[i]);}
        for(int i:second.rows) {remaining[i]->state=Lost;new_lost.push_back(remaining[i]);}
        std::vector<Detection> unused;
        for(int j:first.cols) unused.push_back(high_dets[j]);
        auto third=assign(unconfirmed,unused,.7);
        for(auto [i,j]:third.pairs) {unconfirmed[i]->update(unused[j],frame);active.push_back(unconfirmed[i]);}
        for(int i:third.rows) {unconfirmed[i]->state=Removed;new_removed.push_back(unconfirmed[i]);}
        for(int j:third.cols) if(double(unused[j].score)>=new_score)
            active.push_back(std::make_shared<Track>(unused[j],frame,next_id++));
        for(auto& t:lost) if(frame-t->frame>max_lost) {t->state=Removed;new_removed.push_back(t);}
        tracked.erase(std::remove_if(tracked.begin(),tracked.end(),[](const Ptr& t){return t->state!=Tracked;}),tracked.end());
        tracked=join(join(tracked,active),refind);
        lost=sub(lost,tracked);lost.insert(lost.end(),new_lost.begin(),new_lost.end());
        // Upstream removes against the PREVIOUS removed list, not new_removed.
        lost=sub(lost,removed);
        std::unordered_set<int> drop_tracked,drop_lost;
        for(auto& a:tracked) for(auto& b:lost) if(distance(a->bounds(),b->bounds())<.15f) {
            if(a->frame-a->start>b->frame-b->start) drop_lost.insert(b->id);
            else drop_tracked.insert(a->id);
        }
        tracked.erase(std::remove_if(tracked.begin(),tracked.end(),[&](const Ptr& t){return drop_tracked.count(t->id);}),tracked.end());
        lost.erase(std::remove_if(lost.begin(),lost.end(),[&](const Ptr& t){return drop_lost.count(t->id);}),lost.end());
        removed.insert(removed.end(),new_removed.begin(),new_removed.end());
        if(removed.size()>1000) removed.erase(removed.begin(),removed.end()-999);
    }
};
thread_local std::string error;
}
extern "C" {
const char* dist_error() { return error.c_str(); }
void* dist_create(double fps,double high,double low,double ns,double match,int buffer) {
    try { return new Tracker(fps,high,low,ns,match,buffer); }
    catch(const std::exception& e) {error=e.what();return nullptr;}
}
void dist_destroy(void* p) { delete static_cast<Tracker*>(p); }
int dist_update(void* p,const float* boxes,int n,const double* warp,int* pairs,int capacity) {
    try {
        if(!p || n<0 || (n && !boxes) || !warp || !pairs) throw std::runtime_error("Invalid Dist input");
        for(int i=0;i<6;++i) if(!std::isfinite(warp[i])) throw std::runtime_error("Non-finite GMC");
        auto& tracker=*static_cast<Tracker*>(p);tracker.update(boxes,n,warp);int count=0;
        for(auto& t:tracker.tracked) if(t->confirmed && t->frame==tracker.frame) {
            if(count>=capacity) throw std::runtime_error("Output capacity exceeded");
            pairs[2*count]=t->id;pairs[2*count+1]=t->index;++count;
        }
        return count;
    } catch(const std::exception& e) {error=e.what();return -1;}
}
// Debug-only snapshot: list, ID, state, confirmed, frame, start, observation, mean, covariance.
int dist_snapshot(void* p,double* rows,int capacity) {
    auto& t=*static_cast<Tracker*>(p);int count=0;
    for(int kind=0;kind<2;++kind) for(auto& s:(kind?t.lost:t.tracked)) {
        if(count>=capacity) return -1;
        double* row=rows+count*79;
        row[0]=kind;row[1]=s->id;row[2]=s->state;row[3]=s->confirmed;
        row[4]=s->frame;row[5]=s->start;row[6]=s->index;
        std::copy(s->mean.values.begin(),s->mean.values.end(),row+7);
        std::copy(s->covariance.values.begin(),s->covariance.values.end(),row+15);++count;
    }
    return count;
}
}
