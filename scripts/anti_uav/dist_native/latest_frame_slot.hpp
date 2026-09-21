#pragma once
#include <cstdint>
#include <utility>

// Caller holds the pipeline mutex. Only unclaimed frames may be replaced.
template<typename T> struct LatestFrameSlot {
    T value{};
    uint64_t published=0,replaced=0,taken=0;
    bool ready() const {return bool(value);}
    void publish(T next) {
        if(value) ++replaced;
        value=std::move(next);++published;
    }
    T take() {
        if(!value) return {};
        ++taken;return std::exchange(value,T{});
    }
};
