#include "sector.h"

#include "hamiltonian.h"
#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ed {

namespace {

/**
 * @detail Groups the columns of a sector into orbits of the little group.  The
 *         action is a permutation, so a plain walk marks each orbit once; the
 *         result is a list of orbits, each a list of column indices.
 * @short Orbits of the little group on the sector columns.
 */
std::vector<std::vector<int>> column_orbits(const std::vector<PhasedPermutation>& actions,
                                            int count)
{
    std::vector<char> seen(count, 0);
    std::vector<std::vector<int>> orbits;
    std::vector<int> stack;
    for (int start = 0; start < count; ++start) {
        if (seen[start]) continue;
        std::vector<int> orbit;
        stack.assign(1, start);
        seen[start] = 1;
        while (!stack.empty()) {
            const int x = stack.back();
            stack.pop_back();
            orbit.push_back(x);
            for (const PhasedPermutation& action : actions) {
                const int next = action.act[x];
                if (!seen[next]) {
                    seen[next] = 1;
                    stack.push_back(next);
                }
            }
        }
        std::sort(orbit.begin(), orbit.end());
        orbits.push_back(orbit);
    }
    return orbits;
}

}  // namespace

PhasedPermutation sector_action(const SparseColumns& columns, const Basis& basis,
                                const MonomialAction& action)
{
    const int n = basis.size();
    // state -> column of the sector; every state belongs to exactly one column
    std::vector<int> column_of(n, -1);
    for (int c = 0; c < columns.columns(); ++c)
        for (int a = columns.indptr[c]; a < columns.indptr[c + 1]; ++a)
            column_of[columns.indices[a]] = c;

    PhasedPermutation out;
    out.act.resize(columns.columns());
    out.phase.resize(columns.columns());
    for (int c = 0; c < columns.columns(); ++c) {
        const int probe = columns.indices[columns.indptr[c]];
        const Complex probe_amplitude = columns.data[columns.indptr[c]];
        const auto moved = apply(basis, action, basis.state(probe));
        const int image = basis.index(moved.first);
        if (image < 0 || column_of[image] < 0)
            throw std::runtime_error("the operation leaves the momentum sector");
        const int target = column_of[image];
        // coefficient of the image in its own column
        Complex image_amplitude{0.0, 0.0};
        for (int a = columns.indptr[target]; a < columns.indptr[target + 1]; ++a)
            if (columns.indices[a] == image) image_amplitude = columns.data[a];
        if (std::abs(image_amplitude) < 1e-14)
            throw std::runtime_error("the image of a probe has no amplitude");
        out.act[c] = target;
        out.phase[c] = probe_amplitude * moved.second / image_amplitude;
    }
    return out;
}

SectorVectors isotypic_basis(const std::vector<PhasedPermutation>& actions,
                             const std::vector<Complex>& characters,
                             int dim, int column_count, int threads)
{
    const int groups = static_cast<int>(characters.size());
    if (static_cast<int>(actions.size()) != groups)
        throw std::invalid_argument("one action per character is required");
    const std::vector<std::vector<int>> orbits = column_orbits(actions, column_count);

    // the orbits are independent, so they are spread over the workers; each worker
    // keeps its own copy of the orbit-to-index scratch, and only the current
    // orbit's entries are written -- filling the whole sector per orbit is what
    // this loop used to cost
    std::vector<std::vector<std::vector<std::pair<int, Complex>>>> produced(orbits.size());
    parallel_ranges(static_cast<int>(orbits.size()),
                    [&](int begin, int end) {
        std::vector<int> local(column_count, -1);
        std::vector<Complex> candidate;
        for (int index = begin; index < end; ++index) {
            const std::vector<int>& orbit = orbits[index];
            const int size = static_cast<int>(orbit.size());
            for (int i = 0; i < size; ++i) local[orbit[i]] = i;

            // the rank the character predicts on this orbit, from the trace of the
            // projector: it is what the orthonormalisation below has to reproduce
            Complex trace{0.0, 0.0};
            for (int g = 0; g < groups; ++g) {
                Complex fixed{0.0, 0.0};
                for (int i = 0; i < size; ++i)
                    if (local[actions[g].act[orbit[i]]] == i) fixed += actions[g].phase[orbit[i]];
                trace += std::conj(characters[g]) * fixed;
            }
            const int expected = static_cast<int>(std::lround(
                (static_cast<double>(dim) / static_cast<double>(groups)) * trace.real()));

            // the projected basis vectors of this orbit, orthonormalized one at a
            // time: the image of the projector is spanned by them, so this gives an
            // orthonormal basis of the isotypic component of the orbit
            std::vector<std::vector<std::pair<int, Complex>>> accepted;
            for (int x = 0; x < size && static_cast<int>(accepted.size()) < expected; ++x) {
                candidate.assign(size, Complex{0.0, 0.0});
                for (int g = 0; g < groups; ++g) {
                    const int to = local[actions[g].act[orbit[x]]];
                    candidate[to] += std::conj(characters[g]) * actions[g].phase[orbit[x]]
                                     * (static_cast<double>(dim) / static_cast<double>(groups));
                }
                // two passes of modified Gram-Schmidt over the accepted supports; a
                // single pass loses orthogonality as soon as the orbit is larger
                // than a handful of columns
                for (int pass = 0; pass < 2; ++pass)
                    for (const auto& keep : accepted) {
                        Complex overlap{0.0, 0.0};
                        for (const auto& entry : keep)
                            overlap += std::conj(entry.second) * candidate[entry.first];
                        for (const auto& entry : keep)
                            candidate[entry.first] -= overlap * entry.second;
                    }
                double length = 0.0;
                for (const Complex value : candidate) length += std::norm(value);
                length = std::sqrt(length);
                if (length < 1e-8) continue;        // dependent on what is already in
                std::vector<std::pair<int, Complex>> vector;
                for (int i = 0; i < size; ++i) {
                    if (std::abs(candidate[i]) < 1e-14) continue;
                    vector.emplace_back(i, candidate[i] / length);
                }
                accepted.push_back(std::move(vector));
            }
            if (static_cast<int>(accepted.size()) != expected)
                throw std::runtime_error(
                    "the isotypic component of an orbit has the wrong dimension: the "
                    "character table and the action on the sector disagree");

            // one column per accepted vector: the sector columns are the orbit's
            // own indices, which is what block_matrix expects
            for (const auto& vector : accepted) {
                std::vector<std::pair<int, Complex>> column;
                column.reserve(vector.size());
                for (const auto& entry : vector)
                    column.emplace_back(orbit[entry.first], entry.second);
                produced[index].push_back(std::move(column));
            }
            for (int i = 0; i < size; ++i) local[orbit[i]] = -1;   // leave the scratch clean
        }
    }, threads);

    SectorVectors out;
    out.start.assign(1, 0);
    for (const auto& orbit_vectors : produced)
        for (const auto& column : orbit_vectors) {
            for (const auto& entry : column) {
                out.columns.push_back(entry.first);
                out.data.push_back(entry.second);
            }
            out.start.push_back(static_cast<int>(out.columns.size()));
        }
    return out;
}

SectorVectors identity_vectors(int column_count)
{
    SectorVectors out;
    out.start.resize(static_cast<std::size_t>(column_count) + 1);
    out.columns.resize(column_count);
    out.data.assign(column_count, Complex{1.0, 0.0});
    for (int c = 0; c < column_count; ++c) {
        out.start[c] = c;
        out.columns[c] = c;
    }
    out.start[column_count] = column_count;
    return out;
}

namespace {

/**
 * @detail Sums into a keyed slot with open addressing and linear probing.  The
 *         point is locality: a column of the product produces a few thousand
 *         entries whose keys are scattered over the whole basis, and a table of a
 *         few hundred kilobytes answers every one of them from cache, where the
 *         array indexed by the key itself would miss on each.  Slots are reset
 *         through the list of the ones actually used, so a column costs its own
 *         size and not the size of the table.
 * @short A small cache-resident accumulator.
 */
class Accumulator {
public:
    /**
     * @param capacity how many entries a column can hold, before probing.
     * @detail The table is twice that, rounded up to a power of two, which keeps
     *         the probe short without a load factor argument.
     */
    explicit Accumulator(int capacity)
    {
        int slots = 16;
        while (slots < 2 * capacity) slots <<= 1;
        keys_.assign(static_cast<std::size_t>(slots), -1);
        values_.assign(static_cast<std::size_t>(slots), Complex{0.0, 0.0});
        mask_ = slots - 1;
    }

    /** Forget every entry, touching only the slots that were used. */
    void clear()
    {
        for (const int slot : used_) keys_[slot] = -1;
        used_.clear();
    }

    /** Add *value* to the slot of *key*, creating it if this is the first time. */
    void add(int key, Complex value)
    {
        std::size_t slot = static_cast<std::size_t>(key) * 0x9e3779b1u & mask_;
        while (true) {
            if (keys_[slot] == key) { values_[slot] += value; return; }
            if (keys_[slot] < 0) {
                keys_[slot] = key;
                values_[slot] = value;
                used_.push_back(static_cast<int>(slot));
                return;
            }
            slot = (slot + 1) & mask_;
        }
    }

    /**
     * @param out overwritten with the entries, ascending by key.
     * @detail The caller wants the column's entries in ascending order, and the
     *         distinct keys are far fewer than the images that produced them, so
     *         sorting here is cheap where sorting the images was not.
     */
    void take(std::vector<std::pair<int, Complex>>& out)
    {
        out.clear();
        out.reserve(used_.size());
        for (const int slot : used_) out.emplace_back(keys_[static_cast<std::size_t>(slot)],
                                                      values_[static_cast<std::size_t>(slot)]);
        std::sort(out.begin(), out.end(),
                  [](const auto& left, const auto& right) { return left.first < right.first; });
    }

private:
    std::vector<int> keys_;
    std::vector<Complex> values_;
    std::vector<int> used_;
    std::size_t mask_ = 0;
};

}  // namespace

SparseMatrix sector_hamiltonian(const Basis& basis, const std::vector<CouplingTerm>& couplings,
                                const SparseColumns& columns, int threads)
{
    const int n = basis.size();
    const int count = columns.columns();

    // Where a state sits in the momentum basis.  The columns are translation
    // orbits, so every state is in exactly one of them: the map is one entry per
    // state, and reading it is the only random access left in the product.  It
    // replaces the three arrays that used to hold the same information apart,
    // which made one lookup touch two unrelated cache lines.
    struct Placement {
        int column;
        Complex weight;
    };
    std::vector<Placement> placement(static_cast<std::size_t>(n));
    for (int c = 0; c < count; ++c)
        for (int a = columns.indptr[c]; a < columns.indptr[c + 1]; ++a)
            placement[columns.indices[a]] = Placement{c, columns.data[a]};

    // The terms are read into their tables once per call.
    std::vector<PreparedTerm> prepared;
    prepared.reserve(couplings.size());
    for (const CouplingTerm& term : couplings) prepared.push_back(prepare_term(basis, term));

    // One column of the product at a time: H is applied to the column's states,
    // and the images are accumulated in a local list and merged in place.  The
    // list holds one entry per image of one column -- a few thousand -- so it
    // lives in cache, and no accumulator the size of the basis is touched at all;
    // that is the difference between a random access per image and a local one.
    // the widest column decides the size of the tables once, so no worker has to
    // grow anything while it works
    int widest = 0;
    for (int c = 0; c < count; ++c)
        widest = std::max(widest, columns.indptr[c + 1] - columns.indptr[c]);
    const int capacity = widest * static_cast<int>(prepared.size()) + 1;

    std::vector<std::vector<std::pair<int, Complex>>> produced(static_cast<std::size_t>(count));
    parallel_ranges(count, [&](int begin, int end) {
        Accumulator images(capacity);
        Accumulator scattered(capacity);
        std::vector<std::pair<int, Complex>> merged;
        std::vector<int> digits;
        for (int c = begin; c < end; ++c) {
            images.clear();
            for (int a = columns.indptr[c]; a < columns.indptr[c + 1]; ++a) {
                const int state = columns.indices[a];
                const Complex weight = columns.data[a];
                basis.unpack_into(basis.state(state), digits);
                merged.clear();
                for (const PreparedTerm& term : prepared)
                    apply_prepared(basis, term, basis.state(state), digits, merged);
                for (const auto& entry : merged) images.add(entry.first, weight * entry.second);
            }
            // the images of the column, scattered over the columns that hold them:
            // that is B^T applied to the result of H
            images.take(merged);
            scattered.clear();
            for (const auto& entry : merged) {
                if (std::abs(entry.second) <= 1e-14) continue;
                const Placement& where = placement[static_cast<std::size_t>(entry.first)];
                scattered.add(where.column, std::conj(where.weight) * entry.second);
            }
            std::vector<std::pair<int, Complex>>& column = produced[c];
            scattered.take(column);
            column.erase(std::remove_if(column.begin(), column.end(),
                                        [](const auto& entry) {
                                            return std::abs(entry.second) <= 1e-14;
                                        }),
                         column.end());
        }
    }, threads);

    SparseMatrix by_column;
    by_column.rows = count;
    by_column.cols = count;
    by_column.indptr.assign(static_cast<std::size_t>(count) + 1, 0);
    for (int c = 0; c < count; ++c) {
        for (const auto& entry : produced[c]) {
            by_column.indices.push_back(entry.first);
            by_column.data.push_back(entry.second);
        }
        by_column.indptr[c + 1] = static_cast<int>(by_column.indices.size());
    }
    return transpose(by_column);
}

SparseMatrix block_matrix(const SparseMatrix& matrix, const SectorVectors& vectors,
                          int threads)
{
    const int m = vectors.vectors();
    SparseMatrix by_column;
    by_column.rows = m;
    by_column.cols = m;
    by_column.indptr.assign(static_cast<std::size_t>(m) + 1, 0);
    if (m == 0) return by_column;

    // V^T, walked by sector column: the block rows whose vector contains that
    // column.  A vector lives on one orbit, so this stays as sparse as V.
    std::vector<int> start(static_cast<std::size_t>(matrix.cols) + 1, 0);
    for (const int column : vectors.columns) ++start[column + 1];
    for (int a = 0; a < matrix.cols; ++a) start[a + 1] += start[a];
    std::vector<int> row_of(vectors.columns.size());
    std::vector<Complex> value_of(vectors.columns.size());
    std::vector<int> cursor(start.begin(), start.end() - 1);
    for (int c = 0; c < m; ++c)
        for (int a = vectors.start[c]; a < vectors.start[c + 1]; ++a) {
            const int at = cursor[vectors.columns[a]]++;
            row_of[at] = c;
            value_of[at] = std::conj(vectors.data[a]);
        }

    // V^T H V, one column at a time: H applied to the column of V stays sparse,
    // and every entry of it is scattered over the few block rows that contain its
    // sector row.  The product walks the columns of H, so H is transposed once;
    // the columns of the block are independent and are spread over the workers.
    const SparseMatrix transposed = transpose(matrix);
    std::vector<std::vector<std::pair<int, Complex>>> produced(static_cast<std::size_t>(m));
    parallel_ranges(m, [&](int begin, int end) {
        std::vector<std::pair<int, Complex>> applied;
        std::vector<std::pair<int, Complex>> terms;
        for (int c = begin; c < end; ++c) {
            applied.clear();
            for (int a = vectors.start[c]; a < vectors.start[c + 1]; ++a) {
                const int source = vectors.columns[a];
                const Complex weight = vectors.data[a];
                for (int b = transposed.indptr[source]; b < transposed.indptr[source + 1]; ++b)
                    applied.emplace_back(transposed.indices[b], transposed.data[b] * weight);
            }
            std::sort(applied.begin(), applied.end(),
                      [](const auto& left, const auto& right) { return left.first < right.first; });
            terms.clear();
            for (std::size_t i = 0; i < applied.size();) {
                const int row = applied[i].first;
                Complex sum{0.0, 0.0};
                std::size_t j = i;
                while (j < applied.size() && applied[j].first == row) {
                    sum += applied[j].second;
                    ++j;
                }
                i = j;
                if (std::abs(sum) < 1e-14) continue;
                for (int t = start[row]; t < start[row + 1]; ++t)
                    terms.emplace_back(row_of[t], value_of[t] * sum);
            }
            std::sort(terms.begin(), terms.end(),
                      [](const auto& left, const auto& right) { return left.first < right.first; });
            std::vector<std::pair<int, Complex>>& column = produced[c];
            for (std::size_t i = 0; i < terms.size();) {
                const int row = terms[i].first;
                Complex sum{0.0, 0.0};
                std::size_t j = i;
                while (j < terms.size() && terms[j].first == row) {
                    sum += terms[j].second;
                    ++j;
                }
                i = j;
                if (std::abs(sum) < 1e-14) continue;
                column.emplace_back(row, sum);
            }
        }
    }, threads);

    for (int c = 0; c < m; ++c) {
        for (const auto& entry : produced[c]) {
            by_column.indices.push_back(entry.first);
            by_column.data.push_back(entry.second);
        }
        by_column.indptr[c + 1] = static_cast<int>(by_column.indices.size());
    }
    return transpose(by_column);
}

}  // namespace ed
