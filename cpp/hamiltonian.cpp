#include "hamiltonian.h"

#include <algorithm>
#include <cmath>
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

PreparedTerm prepare_term(const Basis& basis, const CouplingTerm& term)
{
    PreparedTerm prepared;
    prepared.term = &term;
    const int bits = basis.bits();
    const int sites = basis.sites();
    const int local = basis.local_dim();
    for (std::size_t position = 0; position < term.sites.size(); ++position) {
        const int site = term.sites[position];
        const LocalTerm& one_site = term.local[position];
        std::vector<int> image(static_cast<std::size_t>(local), -1);
        std::vector<Complex> phase(static_cast<std::size_t>(local), Complex{0.0, 0.0});
        for (std::size_t entry = 0; entry < one_site.in.size(); ++entry) {
            const int digit = one_site.in[entry];
            if (image[digit] >= 0) {          // one digit, two images: not monomial
                prepared.monomial = false;
                return prepared;
            }
            image[digit] = one_site.out[entry];
            phase[digit] = one_site.coeff[entry];
        }
        prepared.site.push_back(site);
        prepared.shift.push_back(bits * (sites - 1 - site));
        prepared.image.push_back(std::move(image));
        prepared.phase.push_back(std::move(phase));
    }
    prepared.monomial = true;
    return prepared;
}

void apply_prepared(const Basis& basis, const PreparedTerm& prepared, std::uint64_t word,
                    const std::vector<int>& digits, std::vector<std::pair<int, Complex>>& out)
{
    if (!prepared.monomial) {
        apply_term(basis, *prepared.term, word, digits, out);
        return;
    }
    const int bits = basis.bits();
    std::uint64_t image = word;
    Complex coefficient{1.0, 0.0};
    for (std::size_t position = 0; position < prepared.site.size(); ++position) {
        const int digit = digits[prepared.site[position]];
        const int target = prepared.image[position][digit];
        if (target < 0) return;               // the term annihilates this state
        const int shift = prepared.shift[position];
        const std::uint64_t mask = ((1ull << bits) - 1ull) << shift;
        image = (image & ~mask) | (static_cast<std::uint64_t>(target) << shift);
        coefficient *= prepared.phase[position][digit];
    }
    const int target = basis.index(image);
    if (target >= 0) out.emplace_back(target, coefficient);
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

std::vector<Complex> multiply(const SparseMatrix& matrix, const std::vector<Complex>& vector,
                              Workers* workers)
{
    if (static_cast<int>(vector.size()) != matrix.cols)
        throw std::invalid_argument("mat-vec: the vector has the wrong size");
    std::vector<Complex> out(static_cast<std::size_t>(matrix.rows), Complex{0.0, 0.0});
    const auto rows = [&](int, int begin, int end) {
        for (int i = begin; i < end; ++i) {
            Complex sum{0.0, 0.0};
            for (int a = matrix.indptr[i]; a < matrix.indptr[i + 1]; ++a)
                sum += matrix.data[a] * vector[matrix.indices[a]];
            out[i] = sum;
        }
    };
    // the rows are only worth spreading when the matrix has enough of them to pay
    // for the handshake
    if (workers == nullptr || matrix.nonzeros() < (1 << 16)) rows(0, 0, matrix.rows);
    else workers->run(matrix.rows, rows);
    return out;
}

}  // namespace ed
