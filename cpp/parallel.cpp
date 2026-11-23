#include "parallel.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>

namespace ed {

namespace {

// one mutex and two condition variables for the pool: a phase is short, so the
// handshake between the caller and the workers has to be short as well
std::mutex pool_mutex;
std::condition_variable pool_wake;
std::condition_variable pool_done;

}  // namespace

int worker_count(int tasks, int threads)
{
    if (tasks <= 1) return 1;
    const int hardware = static_cast<int>(std::thread::hardware_concurrency());
    const int wanted = threads > 0 ? threads : (hardware > 0 ? hardware : 1);
    return std::max(1, std::min(wanted, tasks));
}

void parallel_ranges(int tasks, const std::function<void(int, int)>& body, int threads)
{
    const int workers = worker_count(tasks, threads);
    if (workers <= 1) {
        body(0, tasks);
        return;
    }
    // contiguous ranges, one per worker: the pieces are equal-sized, so there is
    // nothing for a work queue to balance
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(workers));
    for (int worker = 0; worker < workers; ++worker) {
        const int begin = static_cast<int>(static_cast<long long>(tasks) * worker / workers);
        const int end = static_cast<int>(static_cast<long long>(tasks) * (worker + 1) / workers);
        pool.emplace_back([&body, begin, end]() { body(begin, end); });
    }
    for (std::thread& thread : pool) thread.join();
}

Workers::Workers(int threads)
    : mutex_(&pool_mutex), wake_(&pool_wake), done_(&pool_done)
{
    const int wanted = worker_count(1 << 30, threads);   // the hardware thread count
    for (int index = 1; index < wanted; ++index) threads_.emplace_back([this, index]() { work(index); });
}

Workers::~Workers()
{
    {
        std::lock_guard<std::mutex> lock(*mutex_);
        stopping_ = true;
    }
    wake_->notify_all();
    for (std::thread& thread : threads_) thread.join();
}

void Workers::run(int tasks, const std::function<void(int, int, int)>& body)
{
    const int workers = std::min(size(), std::max(1, tasks));
    if (workers <= 1) {
        body(0, 0, tasks);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(*mutex_);
        body_ = &body;
        tasks_ = tasks;
        ranges_ = workers;                            // fixed until the next generation
        pending_ = workers - 1;                       // the caller takes range zero
        ++generation_;
    }
    wake_->notify_all();
    body(0, 0, static_cast<int>(static_cast<long long>(tasks) / workers));
    {
        std::unique_lock<std::mutex> lock(*mutex_);
        done_->wait(lock, [this]() { return pending_ == 0; });
        body_ = nullptr;
    }
}

void Workers::work(int index)
{
    int served = 0;
    while (true) {
        int begin = 0;
        int end = 0;
        {
            std::unique_lock<std::mutex> lock(*mutex_);
            wake_->wait(lock, [&]() { return stopping_ || generation_ != served; });
            if (stopping_) return;
            served = generation_;
            // ranges_ is read, never recomputed from a count the other workers are
            // changing: that is what made a late worker take a different split and
            // skip its decrement
            if (index >= ranges_) continue;           // fewer ranges than workers
            begin = static_cast<int>(static_cast<long long>(tasks_) * index / ranges_);
            end = static_cast<int>(static_cast<long long>(tasks_) * (index + 1) / ranges_);
        }
        // the body runs outside the lock, otherwise the workers would take turns
        (*body_)(index, begin, end);
        {
            std::lock_guard<std::mutex> lock(*mutex_);
            --pending_;
        }
        done_->notify_one();
    }
}

}  // namespace ed
