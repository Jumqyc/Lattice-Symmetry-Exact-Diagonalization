#pragma once

#include <functional>

namespace ed {

/**
 * @param tasks how many independent pieces of work there are.
 * @param threads how many workers to use; 0 means one per hardware thread.
 * @detail Clamped to the number of tasks, and to the hardware when the caller
 *         asks for everything.  Nothing is created when there is only one piece
 *         of work, so a small call costs a branch and nothing else.
 * @short How many workers a call would actually use.
 */
int worker_count(int tasks, int threads = 0);

/**
 * @param tasks how many independent pieces of work there are.
 * @param body called once per worker with the half-open range it owns.
 * @param threads how many workers to use; 0 means one per hardware thread.
 * @detail Every worker builds its own scratch inside the body, so the loop can
 *         be over the columns of a product without any shared mutable state.  The
 *         ranges are contiguous and handed out in order, which keeps the writes
 *         cache-friendly; there is no dynamic scheduling, because the pieces are
 *         all the same size.
 * @short Run worker(body) on contiguous ranges covering [0, tasks).
 */
void parallel_ranges(int tasks, const std::function<void(int, int)>& body,
                     int threads = 0);

}  // namespace ed
