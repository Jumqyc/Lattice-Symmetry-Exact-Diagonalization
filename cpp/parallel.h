#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ed {

/**
 * @param tasks how many independent pieces of work there are.
 * @param threads how many workers to use; 0 means one per hardware thread.
 * @detail Clamped to the number of tasks, and to the hardware when the caller
 *         asks for everything.
 * @short How many workers a call would actually use.
 */
int worker_count(int tasks, int threads = 0);

/**
 * @param tasks how many independent pieces of work there are.
 * @param body called once per worker with the half-open range it owns.
 * @param threads how many workers to use; 0 means one per hardware thread.
 * @detail Every worker builds its own scratch inside the body, so the loop can be
 *         over the columns of a product without any shared mutable state.  The
 *         ranges are contiguous and handed out in order, which keeps the writes
 *         cache-friendly.
 * @short Run body on contiguous ranges covering [0, tasks), in fresh threads.
 */
void parallel_ranges(int tasks, const std::function<void(int, int)>& body,
                     int threads = 0);

/**
 * @detail A recurrence is a long sequence of short parallel phases -- a dot
 *         product, an axpy, the overlaps with the vectors kept so far -- and a
 *         phase is far too short to pay for creating threads: at a sector of ten
 *         thousand columns a phase is tens of microseconds and a fresh thread is
 *         about the same.  So the threads are created once, here, and the phases
 *         hand them work through a barrier.  The worker's body must not throw and
 *         must not call run() itself; ranges are contiguous, as in
 *         parallel_ranges.
 * @short A fixed set of threads that runs short parallel phases back to back.
 */
class Workers {
public:
    /**
     * @param threads how many threads to use; 0 means one per hardware thread.
     * @detail One of the threads is the caller's, so a pool of *threads* runs
     *         *threads* ranges at a time and no thread is left idle.
     */
    explicit Workers(int threads = 0);
    ~Workers();

    Workers(const Workers&) = delete;
    Workers& operator=(const Workers&) = delete;

    /** The number of ranges a call runs at once. */
    int size() const { return static_cast<int>(threads_.size()) + 1; }

    /**
     * @param tasks how many independent pieces of work there are.
     * @param body called once per range served, with its half-open bounds.
     * @detail Blocks until every range is done, so consecutive calls see each
     *         other's writes: that is what makes a dot product followed by an
     *         axpy safe with the same pool.
     */
    void run(int tasks, const std::function<void(int, int, int)>& body);

private:
    void work(int index);
    std::vector<std::thread> threads_;
    std::mutex* mutex_;
    std::condition_variable* wake_;
    std::condition_variable* done_;
    const std::function<void(int, int, int)>* body_ = nullptr;
    int tasks_ = 0;
    int generation_ = 0;
    int ranges_ = 0;      // how many ranges this generation has; fixed for it
    int pending_ = 0;     // how many of them are still being worked on
    bool stopping_ = false;
};

}  // namespace ed
