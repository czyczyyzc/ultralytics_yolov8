#include "latest_frame_slot.hpp"
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

void check(bool value) {if(!value) throw std::runtime_error("latest frame slot test failed");}
int main() {
    using Ptr=std::shared_ptr<const int>;
    LatestFrameSlot<Ptr> slot;
    check(!slot.take() && slot.taken==0);
    slot.publish(std::make_shared<const int>(1));
    auto held=slot.take();
    slot.publish(std::make_shared<const int>(2));
    slot.publish(std::make_shared<const int>(3));
    check(*held==1 && *slot.take()==3 && slot.replaced==1 && !slot.ready());
    check(slot.published==slot.replaced+slot.taken);
    LatestFrameSlot<Ptr> concurrent;
    std::mutex mutex;std::condition_variable cv;bool done=false;
    std::thread producer([&] {
        for(int i=1;i<=100000;++i) {
            auto frame=std::make_shared<const int>(i);
            {std::lock_guard<std::mutex> lock(mutex);concurrent.publish(std::move(frame));}
            cv.notify_one();
        }
        {std::lock_guard<std::mutex> lock(mutex);done=true;}cv.notify_one();
    });
    int last=0;bool ordered=true;
    while(true) {
        Ptr frame;
        {std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return done || concurrent.ready();});
         if(!concurrent.ready()) break;frame=concurrent.take();}
        if(*frame<=last) ordered=false;
        last=*frame;std::this_thread::yield();
        if(*frame!=last) ordered=false;
    }
    producer.join();
    check(ordered && last==100000 && concurrent.published==100000);
    check(!concurrent.ready() && concurrent.published==concurrent.taken+concurrent.replaced);
    std::cout<<"passed replacement, retained-frame ownership and 100000 concurrent publications\n";
}
