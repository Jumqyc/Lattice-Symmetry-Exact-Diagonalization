#include "sector.h"

#include "hamiltonian.h"

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
                             int dim, int column_count)
{
    const int groups = static_cast<int>(characters.size());
    if (static_cast<int>(actions.size()) != groups)
        throw std::invalid_argument("one action per character is required");
    const std::vector<std::vector<int>> orbits = column_orbits(actions, column_count);

    SectorVectors out;
    out.start.assign(1, 0);
    std::vector<Complex> candidate;             // one orbit, never the sector
    for (int index = 0; index < static_cast<int>(orbits.size()); ++index) {
        const std::vector<int>& orbit = orbits[index];
        const int size = static_cast<int>(orbit.size());
        std::vector<int> local(column_count, -1);
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
            double norm = 0.0;
            for (const Complex value : candidate) norm += std::norm(value);
            norm = std::sqrt(norm);
            if (norm < 1e-8) continue;          // dependent on what is already in
            std::vector<std::pair<int, Complex>> vector;
            for (int i = 0; i < size; ++i) {
                if (std::abs(candidate[i]) < 1e-14) continue;
                vector.emplace_back(i, candidate[i] / norm);
            }
            accepted.push_back(std::move(vector));
        }
        if (static_cast<int>(accepted.size()) != expected)
            throw std::runtime_error(
                "the isotypic component of an orbit has the wrong dimension: the "
                "character table and the action on the sector disagree");

        for (const auto& vector : accepted) {
            for (const auto& entry : vector) out.columns.push_back(orbit[entry.first]);
            for (const auto& entry : vector) out.data.push_back(entry.second);
            out.start.push_back(static_cast<int>(out.columns.size()));
        }
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

SparseMatrix sector_hamiltonian(const SparseMatrix& hamiltonian, const SparseColumns& columns)
{
    const int n = hamiltonian.rows;
    const int count = columns.columns();
    if (hamiltonian.cols != n) throw std::invalid_argument("the Hamiltonian is not square");

    // both factors are walked by column, so both matrices are transposed once
    const SparseMatrix hamiltonian_t = transpose(hamiltonian);
    SparseMatrix columns_t;                 // rows are states, columns are sector columns
    columns_t.rows = n;
    columns_t.cols = count;
    columns_t.indptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int c = 0; c < count; ++c)
        for (int a = columns.indptr[c]; a < columns.indptr[c + 1]; ++a)
            ++columns_t.indptr[columns.indices[a] + 1];
    for (int i = 0; i < n; ++i) columns_t.indptr[i + 1] += columns_t.indptr[i];
    columns_t.indices.resize(columns.data.size());
    columns_t.data.resize(columns.data.size());
    std::vector<int> cursor(columns_t.indptr.begin(), columns_t.indptr.end() - 1);
    for (int c = 0; c < count; ++c)
        for (int a = columns.indptr[c]; a < columns.indptr[c + 1]; ++a) {
            const int at = cursor[columns.indices[a]]++;
            columns_t.indices[at] = c;
            columns_t.data[at] = columns.data[a];
        }

    // the accumulation below runs column by column, so the result is built in
    // compressed-column form and transposed at the end
    SparseMatrix by_column;
    by_column.rows = count;
    by_column.cols = count;
    by_column.indptr.assign(static_cast<std::size_t>(count) + 1, 0);
    // The entries of one column are scattered into a scratch that has the size
    // of the sector and is reset through a stamp, so no vector of the size of
    // the sector is ever refilled and no list is ever sorted: the work follows
    // the number of nonzeros of the column.  Sorting the entry list instead was
    // both slower here and, for a large sector, the only quadratic term left.
    std::vector<Complex> row(static_cast<std::size_t>(count), Complex{0.0, 0.0});
    std::vector<int> stamp(static_cast<std::size_t>(count), -1);
    std::vector<int> touched;
    std::vector<std::pair<int, Complex>> product;   // H B, column c: (state row, value)
    int colour = 0;
    for (int c = 0; c < count; ++c) {
        // Y = H B, column c
        product.clear();
        for (int a = columns.indptr[c]; a < columns.indptr[c + 1]; ++a) {
            const int state = columns.indices[a];
            const Complex weight = columns.data[a];
            for (int b = hamiltonian_t.indptr[state]; b < hamiltonian_t.indptr[state + 1]; ++b)
                product.emplace_back(hamiltonian_t.indices[b], hamiltonian_t.data[b] * weight);
        }
        // B^T Y, column c
        ++colour;
        touched.clear();
        for (const auto& entry : product)
            for (int b = columns_t.indptr[entry.first]; b < columns_t.indptr[entry.first + 1]; ++b) {
                const int target = columns_t.indices[b];
                if (stamp[target] != colour) {
                    stamp[target] = colour;
                    row[target] = Complex{0.0, 0.0};
                    touched.push_back(target);
                }
                row[target] += std::conj(columns_t.data[b]) * entry.second;
            }
        // the row indices of a column have to be increasing for the result to be
        // a canonical CSR matrix, and there are few of them
        std::sort(touched.begin(), touched.end());
        for (const int target : touched) {
            const Complex value = row[target];
            if (std::abs(value) < 1e-14) continue;
            by_column.indices.push_back(target);
            by_column.data.push_back(value);
        }
        by_column.indptr[c + 1] = static_cast<int>(by_column.indices.size());
    }
    return transpose(by_column);
}

SparseMatrix sector_hamiltonian(const Basis& basis, const std::vector<CouplingTerm>& couplings,
                                const SparseColumns& columns)
{
    const int n = basis.size();
    const int count = columns.columns();

    // state -> the columns that contain it, which is the only structure of B the
    // product needs on the right-hand side
    std::vector<int> start(static_cast<std::size_t>(n) + 1, 0);
    for (int c = 0; c < count; ++c)
        for (int a = columns.indptr[c]; a < columns.indptr[c + 1]; ++a)
            ++start[columns.indices[a] + 1];
    for (int i = 0; i < n; ++i) start[i + 1] += start[i];
    std::vector<int> column_of(start[n]);
    std::vector<Complex> weight_of(start[n]);
    std::vector<int> cursor(start.begin(), start.end() - 1);
    for (int c = 0; c < count; ++c)
        for (int a = columns.indptr[c]; a < columns.indptr[c + 1]; ++a) {
            const int at = cursor[columns.indices[a]]++;
            column_of[at] = c;
            weight_of[at] = columns.data[a];
        }

    // Per output column, Y = H B is accumulated on the states: a scratch of the
    // size of the basis with a stamp, reset through the touched list, so no
    // dense fill and no sort.  The second scratch accumulates B^T Y over the
    // columns of the sector.
    std::vector<Complex> applied(static_cast<std::size_t>(n), Complex{0.0, 0.0});
    std::vector<int> state_stamp(static_cast<std::size_t>(n), -1);
    std::vector<int> state_touched;
    std::vector<Complex> summed(static_cast<std::size_t>(count), Complex{0.0, 0.0});
    std::vector<int> column_stamp(static_cast<std::size_t>(count), -1);
    std::vector<int> column_touched;
    std::vector<std::pair<int, Complex>> images;

    SparseMatrix by_column;
    by_column.rows = count;
    by_column.cols = count;
    by_column.indptr.assign(static_cast<std::size_t>(count) + 1, 0);
    std::vector<int> digits;                        // reused across the whole product
    for (int c = 0; c < count; ++c) {
        state_touched.clear();
        for (int a = columns.indptr[c]; a < columns.indptr[c + 1]; ++a) {
            const int state = columns.indices[a];
            const Complex weight = columns.data[a];
            basis.unpack_into(basis.state(state), digits);
            images.clear();
            for (const CouplingTerm& term : couplings)
                apply_term(basis, term, basis.state(state), digits, images);
            for (const auto& entry : images) {
                const int image = entry.first;
                if (state_stamp[image] != c) {
                    state_stamp[image] = c;
                    applied[image] = Complex{0.0, 0.0};
                    state_touched.push_back(image);
                }
                applied[image] += weight * entry.second;
            }
        }
        column_touched.clear();
        for (const int state : state_touched) {
            const Complex value = applied[state];
            if (std::abs(value) < 1e-14) continue;
            for (int t = start[state]; t < start[state + 1]; ++t) {
                const int column = column_of[t];
                if (column_stamp[column] != c) {
                    column_stamp[column] = c;
                    summed[column] = Complex{0.0, 0.0};
                    column_touched.push_back(column);
                }
                summed[column] += std::conj(weight_of[t]) * value;
            }
        }
        std::sort(column_touched.begin(), column_touched.end());
        for (const int column : column_touched) {
            const Complex value = summed[column];
            if (std::abs(value) < 1e-14) continue;
            by_column.indices.push_back(column);
            by_column.data.push_back(value);
        }
        by_column.indptr[c + 1] = static_cast<int>(by_column.indices.size());
    }
    return transpose(by_column);
}

SparseMatrix block_matrix(const SparseMatrix& matrix, const SectorVectors& vectors)
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
    // and every entry of it is scattered over the few block rows that contain
    // its sector row.  Nothing of the size of the sector is ever allocated.
    // The product walks the columns of H, so H is transposed once: walking its
    // rows instead would accumulate (H^T v).
    const SparseMatrix transposed = transpose(matrix);
    std::vector<std::pair<int, Complex>> applied;
    std::vector<std::pair<int, Complex>> terms;
    for (int c = 0; c < m; ++c) {
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
            by_column.indices.push_back(row);
            by_column.data.push_back(sum);
        }
        by_column.indptr[c + 1] = static_cast<int>(by_column.indices.size());
    }
    return transpose(by_column);
}

}  // namespace ed
