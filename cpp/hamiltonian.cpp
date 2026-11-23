#include "hamiltonian.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>

namespace ed {

namespace {

/**
 * @detail The recursion that replaces, site by site, the local index of the
 *         state by one of the ``out`` values of the term.  It is a plain
 *         function and not a ``std::function``: that one allocates on every
 *         call, and it is called once per (state, term), which at five by five
 *         is a hundred million allocations.
 * @short Carry one branch of the term application to the next site.
 */
void walk_term(const Basis& basis, const CouplingTerm& term, const std::vector<int>& digits,
               std::size_t index, std::uint64_t image, Complex amplitude,
               std::vector<std::pair<int, Complex>>& out)
{
    if (index == term.sites.size()) {
        const int target = basis.index(image);
        if (target >= 0) out.emplace_back(target, amplitude);
        return;
    }
    const int bits = basis.bits();
    const int sites = basis.sites();
    const int site = term.sites[index];
    const LocalTerm& local = term.local[index];
    const int shift = bits * (sites - 1 - site);
    const std::uint64_t mask = ((1ull << bits) - 1ull) << shift;
    for (std::size_t entry = 0; entry < local.in.size(); ++entry) {
        if (local.in[entry] != digits[site]) continue;
        const std::uint64_t shifted = static_cast<std::uint64_t>(local.out[entry]) << shift;
        walk_term(basis, term, digits, index + 1, (image & ~mask) | shifted,
                  amplitude * local.coeff[entry], out);
    }
}

}  // namespace

void apply_term(const Basis& basis, const CouplingTerm& term, std::uint64_t word,
                const std::vector<int>& digits, std::vector<std::pair<int, Complex>>& out)
{
    // One- and two-site terms are almost all of a lattice model, and the
    // recursion costs more than the arithmetic it carries for them: the two
    // cases are written out, the rest falls back to the walk.
    const int bits = basis.bits();
    const int sites = basis.sites();
    if (term.sites.size() == 1) {
        const int site = term.sites[0];
        const LocalTerm& local = term.local[0];
        const int shift = bits * (sites - 1 - site);
        const std::uint64_t mask = ((1ull << bits) - 1ull) << shift;
        for (std::size_t entry = 0; entry < local.in.size(); ++entry) {
            if (local.in[entry] != digits[site]) continue;
            const std::uint64_t shifted = static_cast<std::uint64_t>(local.out[entry]) << shift;
            const int target = basis.index((word & ~mask) | shifted);
            if (target >= 0) out.emplace_back(target, local.coeff[entry]);
        }
        return;
    }
    if (term.sites.size() == 2) {
        const int first = term.sites[0];
        const int second = term.sites[1];
        const LocalTerm& local_first = term.local[0];
        const LocalTerm& local_second = term.local[1];
        const int shift_first = bits * (sites - 1 - first);
        const int shift_second = bits * (sites - 1 - second);
        const std::uint64_t mask_first = ((1ull << bits) - 1ull) << shift_first;
        const std::uint64_t mask_second = ((1ull << bits) - 1ull) << shift_second;
        for (std::size_t a = 0; a < local_first.in.size(); ++a) {
            if (local_first.in[a] != digits[first]) continue;
            const std::uint64_t once =
                ((word & ~mask_first) | (static_cast<std::uint64_t>(local_first.out[a]) << shift_first));
            for (std::size_t b = 0; b < local_second.in.size(); ++b) {
                if (local_second.in[b] != digits[second]) continue;
                const std::uint64_t twice = ((once & ~mask_second)
                                             | (static_cast<std::uint64_t>(local_second.out[b])
                                                << shift_second));
                const int target = basis.index(twice);
                if (target >= 0) out.emplace_back(target, local_first.coeff[a] * local_second.coeff[b]);
            }
        }
        return;
    }
    walk_term(basis, term, digits, 0, word, Complex{1.0, 0.0}, out);
}

SparseMatrix assemble(const Basis& basis, const std::vector<CouplingTerm>& couplings)
{
    const int n = basis.size();
    SparseMatrix matrix;
    matrix.rows = n;
    matrix.cols = n;
    matrix.indptr.assign(n + 1, 0);

    std::vector<std::pair<int, Complex>> column;    // (column index, value) of the row
    std::vector<int> digits;                        // reused: one allocation in total
    for (int i = 0; i < n; ++i) {
        // the state is unpacked once and every term reuses the digits: unpacking
        // per term would touch the whole state once per term
        basis.unpack_into(basis.state(i), digits);
        column.clear();
        for (const CouplingTerm& term : couplings)
            apply_term(basis, term, basis.state(i), digits, column);

        // order the row by column index so that the CSR form is canonical and
        // two implementations can be compared
        std::sort(column.begin(), column.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::size_t k = 0;
        while (k < column.size()) {
            std::size_t next = k;
            Complex sum{0.0, 0.0};
            while (next < column.size() && column[next].first == column[k].first) {
                sum += column[next].second;
                ++next;
            }
            if (std::abs(sum) > 1e-14) {
                matrix.indices.push_back(column[k].first);
                matrix.data.push_back(sum);
            }
            k = next;
        }
        matrix.indptr[i + 1] = static_cast<int>(matrix.indices.size());
    }
    return matrix;
}

SparseMatrix symmetrise(const SparseMatrix& matrix, const Basis& basis,
                        const std::vector<MonomialAction>& actions)
{
    const int n = basis.size();
    const int groups = static_cast<int>(actions.size());
    if (matrix.rows != n || matrix.cols != n)
        throw std::invalid_argument("the matrix does not match the basis");

    // Both paths below work one group element at a time, so the image and the
    // phase of every state are rebuilt per element into two buffers instead of
    // being stored for all of them: the number of `apply` calls is the same, and
    // the memory is O(basis) instead of O(|G| * basis).
    std::vector<int> image(static_cast<std::size_t>(n));
    std::vector<Complex> phase(static_cast<std::size_t>(n));
    const auto map_element = [&](int g, bool with_phase) {
        for (int i = 0; i < n; ++i) {
            const auto moved = apply(basis, actions[g], basis.state(i));
            image[i] = basis.index(moved.first);
            if (image[i] < 0)
                throw std::runtime_error("a group element left the basis; the symmetry is inconsistent");
            if (with_phase) phase[i] = moved.second;
        }
    };

    // fast path: H is already invariant, which is the case whenever the
    // couplings respect the symmetry; averaging it would return it anyway.
    // The check walks every entry once per group element and marks the columns
    // of the target row with a stamp: the images of a sorted row are not sorted,
    // so a merge walk would not do, and scanning the target row per entry costs
    // a factor equal to the number of nonzeros per row.
    bool invariant = true;
    std::vector<int> stamp(static_cast<std::size_t>(n), -1);
    int colour = 0;
    for (int g = 0; g < groups && invariant; ++g) {
        map_element(g, false);
        for (int i = 0; i < n && invariant; ++i) {
            const int source = image[i];
            const auto begin = matrix.indptr[i], end = matrix.indptr[i + 1];
            const auto from = matrix.indptr[source], stop = matrix.indptr[source + 1];
            if (end - begin != stop - from) {
                invariant = false;
                break;
            }
            ++colour;
            for (int b = from; b < stop; ++b) stamp[matrix.indices[b]] = colour;
            for (int a = begin; a < end; ++a) {
                // the image of the entry (i, column) sits at (image[i], image[column])
                if (stamp[image[matrix.indices[a]]] != colour) {
                    invariant = false;
                    break;
                }
            }
        }
    }
    if (invariant) return matrix;

    // general path: accumulate (1/|G|) sum_g U(g) H U(g)^-1
    std::map<std::pair<int, int>, Complex> accumulated;
    for (int g = 0; g < groups; ++g) {
        map_element(g, true);
        for (int i = 0; i < n; ++i) {
            const Complex ratio = phase[i];
            for (int a = matrix.indptr[i]; a < matrix.indptr[i + 1]; ++a) {
                const int j = matrix.indices[a];
                const Complex weight = ratio * std::conj(phase[j]) * matrix.data[a];
                accumulated[{image[i], image[j]}] += weight;
            }
        }
    }

    // the map is ordered by (row, column), so the CSR form follows from one walk
    // over it; looking every pair of states up in turn would be quadratic in the
    // size of the basis
    SparseMatrix out;
    out.rows = n;
    out.cols = n;
    out.indptr.assign(static_cast<std::size_t>(n) + 1, 0);
    int row = 0;
    for (const auto& entry : accumulated) {
        while (row < entry.first.first) out.indptr[++row] = static_cast<int>(out.indices.size());
        const Complex value = entry.second / static_cast<double>(groups);
        if (std::abs(value) < 1e-14) continue;
        out.indices.push_back(entry.first.second);
        out.data.push_back(value);
    }
    while (row < n) out.indptr[++row] = static_cast<int>(out.indices.size());
    return out;
}

SparseMatrix transpose(const SparseMatrix& matrix)
{
    SparseMatrix out;
    out.rows = matrix.cols;
    out.cols = matrix.rows;
    out.indptr.assign(static_cast<std::size_t>(out.rows) + 1, 0);
    for (int a = 0; a < matrix.nonzeros(); ++a) ++out.indptr[matrix.indices[a] + 1];
    for (int i = 0; i < out.rows; ++i) out.indptr[i + 1] += out.indptr[i];
    out.indices.resize(matrix.nonzeros());
    out.data.resize(matrix.nonzeros());
    std::vector<int> cursor(out.indptr.begin(), out.indptr.end() - 1);
    for (int i = 0; i < matrix.rows; ++i)
        for (int a = matrix.indptr[i]; a < matrix.indptr[i + 1]; ++a) {
            const int at = cursor[matrix.indices[a]]++;
            out.indices[at] = i;
            out.data[at] = matrix.data[a];
        }
    return out;
}

std::vector<Complex> multiply(const SparseMatrix& matrix, const std::vector<Complex>& vector)
{
    if (static_cast<int>(vector.size()) != matrix.cols)
        throw std::invalid_argument("mat-vec: the vector has the wrong size");
    std::vector<Complex> out(static_cast<std::size_t>(matrix.rows), Complex{0.0, 0.0});
    for (int i = 0; i < matrix.rows; ++i) {
        Complex sum{0.0, 0.0};
        for (int a = matrix.indptr[i]; a < matrix.indptr[i + 1]; ++a)
            sum += matrix.data[a] * vector[matrix.indices[a]];
        out[i] = sum;
    }
    return out;
}

}  // namespace ed
