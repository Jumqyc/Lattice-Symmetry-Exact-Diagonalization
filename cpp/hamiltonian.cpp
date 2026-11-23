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
               std::vector<std::pair<int, Complex>>& out, bool lookup)
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
                  amplitude * local.coeff[entry], out, lookup);
    }
}

}  // namespace


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
    // the sites are kept ascending so the incremental rank can walk the span the
    // term touches without sorting again in the hot loop
    for (std::size_t i = 0; i + 1 < prepared.site.size(); ++i)
        for (std::size_t j = i + 1; j < prepared.site.size(); ++j)
            if (prepared.site[j] < prepared.site[i]) {
                std::swap(prepared.site[i], prepared.site[j]);
                std::swap(prepared.shift[i], prepared.shift[j]);
                std::swap(prepared.image[i], prepared.image[j]);
                std::swap(prepared.phase[i], prepared.phase[j]);
            }
    prepared.monomial = true;
    return prepared;
}

void apply_prepared(const Basis& basis, const PreparedTerm& prepared, std::uint64_t word,
                    const std::vector<int>& digits, const std::vector<int>& prefix,
                    int source_index, std::vector<std::pair<int, Complex>>& out)
{
    if (!prepared.monomial) {
        std::vector<std::pair<std::uint64_t, Complex>> words;
        walk_term(basis, *prepared.term, digits, 0, word, Complex{1.0, 0.0}, out, basis.has_table());
        return;
    }
    const int bits = basis.bits();
    if (basis.has_table()) {
        // the bit vector answers with two reads, so the image word is built and
        // looked up rather than counted
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
        const int found = basis.index(image);
        if (found >= 0) out.emplace_back(found, coefficient);
        return;
    }
    // no table: the index of the image is the source's index plus the difference
    // the changed digits make, which is a walk over the span they cover
    int sites[8];
    int changed[8];
    const int count = std::min(static_cast<int>(prepared.site.size()), 8);
    Complex coefficient{1.0, 0.0};
    for (int position = 0; position < count; ++position) {
        const int digit = digits[prepared.site[position]];
        const int target = prepared.image[position][digit];
        if (target < 0) return;
        sites[position] = prepared.site[position];
        changed[position] = target;
        coefficient *= prepared.phase[position][digit];
    }
    int image_index = 0;
    if (basis.index_incremental(digits, prefix, sites, changed, count, source_index, image_index))
        out.emplace_back(image_index, coefficient);
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
