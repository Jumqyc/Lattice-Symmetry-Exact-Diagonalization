#include "parallel.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace ed {

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

}  // namespace ed
