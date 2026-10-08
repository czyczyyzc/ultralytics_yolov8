// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "third_party/lap/lapjv.h"

namespace motion {
struct AssignmentResult {
    std::vector<int> columns;
    std::vector<bool> ambiguous;
};

class GlobalAssignment {
    static constexpr int max_checked_nodes=16;
    static constexpr int max_checks_per_frame=8;
    int remaining_checks=max_checks_per_frame;
    std::vector<double> padded;
    std::vector<double*> pointers;
    std::vector<int> x,y;

    std::vector<int> solve(const std::vector<std::vector<double>>& costs,double limit,
                           int blocked_row=-1,int blocked_col=-1) {
        int nr=costs.size(),nc=costs.front().size(),n=nr+nc;
        padded.assign(n*n,limit/2);pointers.resize(n);x.resize(n);y.resize(n);
        for(int i=0;i<n;++i) pointers[i]=padded.data()+i*n;
        for(int i=nr;i<n;++i) for(int j=nc;j<n;++j) pointers[i][j]=0;
        for(int i=0;i<nr;++i) for(int j=0;j<nc;++j)
            pointers[i][j]=(i==blocked_row && j==blocked_col) || costs[i][j]>=limit?100.:costs[i][j];
        if(lapjv_internal(n,pointers.data(),x.data(),y.data())) throw std::runtime_error("Motion LAPJV failed");
        std::vector<int> result(nr,-1);
        for(int i=0;i<nr;++i) if(x[i]>=0 && x[i]<nc && pointers[i][x[i]]<limit) result[i]=x[i];
        return result;
    }
    static double objective(const std::vector<std::vector<double>>& costs,const std::vector<int>& match,double limit) {
        int hits=0;double sum=0;
        for(size_t i=0;i<match.size();++i) if(match[i]>=0) {sum+=costs[i][match[i]];++hits;}
        return sum+(costs.size()+costs.front().size()-2*hits)*limit/2;
    }
    static std::vector<int> owners(const std::vector<int>& match,int nc) {
        std::vector<int> result(nc,-1);
        for(size_t i=0;i<match.size();++i) if(match[i]>=0) result[match[i]]=i;
        return result;
    }
public:
    // frames, admissible edges, components, base solves, alternative solves,
    // locally competitive assignments resolved globally, ambiguous detections, budget abstentions.
    std::array<uint64_t,8> counts{};
    void begin_frame() {remaining_checks=max_checks_per_frame;++counts[0];}
    AssignmentResult assign(const std::vector<std::vector<double>>& cost,double limit,double margin) {
        const int nr=cost.size(),nc=nr?cost.front().size():0;
        AssignmentResult output{std::vector<int>(nr,-1),std::vector<bool>(nc,false)};
        if(!nr || !nc) return output;
        std::vector<std::vector<int>> re(nr),ce(nc);
        for(int i=0;i<nr;++i) {
            if(cost[i].size()!=size_t(nc)) throw std::runtime_error("Ragged assignment matrix");
            for(int j=0;j<nc;++j) {
                if(!std::isfinite(cost[i][j])) throw std::runtime_error("Non-finite assignment cost");
                if(cost[i][j]<limit) {re[i].push_back(j);ce[j].push_back(i);++counts[1];}
            }
        }
        std::vector<bool> seen_rows(nr),seen_cols(nc);
        for(int seed=0;seed<nr;++seed) {
            if(seen_rows[seed] || re[seed].empty()) continue;
            std::vector<int> rows{seed},cols;seen_rows[seed]=true;
            for(size_t at=0;at<rows.size();++at) for(int j:re[rows[at]]) if(!seen_cols[j]) {
                seen_cols[j]=true;cols.push_back(j);
                for(int i:ce[j]) if(!seen_rows[i]) {seen_rows[i]=true;rows.push_back(i);}
            }
            ++counts[2];
            if(rows.size()+cols.size()>max_checked_nodes) {
                for(int j:cols) {output.ambiguous[j]=true;++counts[6];++counts[7];}
                continue;
            }
            std::vector<std::vector<double>> local(rows.size(),std::vector<double>(cols.size(),100.));
            for(size_t i=0;i<rows.size();++i) for(size_t j=0;j<cols.size();++j) local[i][j]=cost[rows[i]][cols[j]];
            auto best=solve(local,limit);++counts[3];
            auto best_owners=owners(best,cols.size());
            double best_cost=objective(local,best,limit);
            std::vector<bool> rejected(cols.size(),false);
            for(size_t i=0;i<best.size();++i) {
                int j=best[i];if(j<0) continue;
                bool competitive=false;
                for(size_t k=0;k<cols.size();++k) if(int(k)!=j && local[i][k]<limit &&
                    std::abs(local[i][k]-local[i][j])<margin) competitive=true;
                for(size_t k=0;k<rows.size();++k) if(k!=i && local[k][j]<limit &&
                    std::abs(local[k][j]-local[i][j])<margin) competitive=true;
                if(re[rows[i]].size()==1 && ce[cols[j]].size()==1) continue;
                if(remaining_checks==0) {
                    std::fill(rejected.begin(),rejected.end(),true);
                    counts[7]+=cols.size();break;
                }
                --remaining_checks;
                auto alternative=solve(local,limit,i,j);++counts[4];
                double gap=objective(local,alternative,limit)-best_cost;
                auto alternate_owners=owners(alternative,cols.size());
                // Dummy permutations and simply refusing a match are not alternative identities.
                bool new_real_edge=false;
                for(size_t k=0;k<alternative.size();++k)
                    if(alternative[k]>=0 && alternative[k]!=best[k]) new_real_edge=true;
                if(new_real_edge && gap<margin) {
                    for(size_t k=0;k<cols.size();++k) if(best_owners[k]!=alternate_owners[k]) rejected[k]=true;
                } else if(competitive) ++counts[5];
            }
            for(size_t j=0;j<cols.size();++j) if(rejected[j]) {output.ambiguous[cols[j]]=true;++counts[6];}
            for(size_t i=0;i<best.size();++i) if(best[i]>=0 && !rejected[best[i]]) output.columns[rows[i]]=cols[best[i]];
        }
        return output;
    }
};
}
