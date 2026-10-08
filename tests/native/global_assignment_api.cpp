#include "global_assignment.hpp"
extern "C" int test_global_assignment(const double* costs,int nr,int nc,double limit,double margin,
                                      int* matches,int* ambiguous,uint64_t* counts) {
    try {
        motion::GlobalAssignment solver;solver.begin_frame();
        std::vector<std::vector<double>> data(nr,std::vector<double>(nc));
        for(int i=0;i<nr;++i) for(int j=0;j<nc;++j) data[i][j]=costs[i*nc+j];
        auto result=solver.assign(data,limit,margin);
        for(int i=0;i<nr;++i) matches[i]=result.columns[i];
        for(int j=0;j<nc;++j) ambiguous[j]=result.ambiguous[j];
        std::copy(solver.counts.begin(),solver.counts.end(),counts);return 0;
    } catch(...) {return -1;}
}
