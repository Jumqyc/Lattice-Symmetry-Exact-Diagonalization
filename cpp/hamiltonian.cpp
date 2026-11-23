#include "hamiltonian.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>

namespace ed {

namespace {

/**
 * @detail Applies one term to a state: every site of the term replaces its
 *         local index by one of the ``out`` values, with the product of the
 *         coefficients.  The recursion walks the term's sites and carries the
 *         packed word, so no temporary state vector is built.
 * @short Add the images of one term to ``out``.
 */
void apply_term(const Basis& basis, const CouplingTerm& term, std::uint64_t word,
                std::vector<std::pair<std::uint64_t, Complex>>& out)
{
    const int bits = basis.bits();
    const int sites = basis.sites();
    const std::vector<int> digits = basis.unpack(word);
    std::function<void(std::size_t, std::uint64_t, Complex)> walk =
        [&](std::size_t index, std::uint64_t image, Complex amplitude) {
            if (index == term.sites.size()) {
                out.emplace_back(image, amplitude);
                return;
            }
            const int site = term.sites[index];
            const LocalTerm& local = term.local[index];
            for (std::size_t entry = 0; entry < local.in.size(); ++entry) {
                if (local.in[entry] != digits[site]) continue;
                const std::uint64_t shifted =
                    static_cast<std::uint64_t>(local.out[entry]) << (bits * (sites - 1 - site));
                const std::uint64_t mask = ((1ull << bits) - 1ull) << (bits * (sites - 1 - site));
                walk(index + 1, (image & ~mask) | shifted, amplitude * local.coeff[entry]);
            }
        };
    walk(0, word, Complex{1.0, 0.0});
}

}  // namespace

SparseMatrix assemble(const Basis& basis, const std::vector<CouplingTerm>& couplings)
{
    const int n = basis.size();
    SparseMatrix matrix;
    matrix.rows = n;
    matrix.cols = n;
    matrix.indptr.assign(n + 1, 0);

    std::vector<std::pair<std::uint64_t, Complex>> images;
    std::vector<std::pair<int, Complex>> column;
    for (int i = 0; i < n; ++i) {
        const std::uint64_t word = basis.state(i);
        images.clear();
        for (const CouplingTerm& term : couplings) apply_term(basis, term, word, images);

        // accumulate equal images, then order the row by column index so that
        // the CSR form is canonical and two implementations can be compared
        column.clear();
        for (const auto& entry : images) {
            const int j = basis.index(entry.first);
            if (j >= 0) column.emplace_back(j, entry.second);
        }
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

    // the image of every state under every element, and the phase it carries
    std::vector<std::vector<int>> image(groups, std::vector<int>(n));
    std::vector<std::vector<Complex>> phase(groups, std::vector<Complex>(n));
    for (int g = 0; g < groups; ++g)
        for (int i = 0; i < n; ++i) {
            const auto moved = apply(basis, actions[g], basis.state(i));
            image[g][i] = basis.index(moved.first);
            phase[g][i] = moved.second;
            if (image[g][i] < 0)
                throw std::runtime_error("a group element left the basis; the symmetry is inconsistent");
        }

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
        for (int i = 0; i < n && invariant; ++i) {
            const int source = image[g][i];
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
                if (stamp[image[g][matrix.indices[a]]] != colour) {
                    invariant = false;
                    break;
                }
            }
        }
    }
    if (invariant) return matrix;

    // general path: accumulate (1/|G|) sum_g U(g) H U(g)^-1
    std::map<std::pair<int, int>, Complex> accumulated;
    for (int g = 0; g < groups; ++g)
        for (int i = 0; i < n; ++i) {
            const Complex ratio = phase[g][i];
            for (int a = matrix.indptr[i]; a < matrix.indptr[i + 1]; ++a) {
                const int j = matrix.indices[a];
                const Complex weight = ratio * std::conj(phase[g][j]) * matrix.data[a];
                accumulated[{image[g][i], image[g][j]}] += weight;
            }
        }

    SparseMatrix out;
    out.rows = n;
    out.cols = n;
    out.indptr.assign(n + 1, 0);
    out.indptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const auto it = accumulated.find({i, j});
            if (it == accumulated.end()) continue;
            const Complex value = it->second / static_cast<double>(groups);
            if (std::abs(value) > 1e-14) {
                out.indices.push_back(j);
                out.data.push_back(value);
            }
        }
        out.indptr[i + 1] = static_cast<int>(out.indices.size());
    }
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
