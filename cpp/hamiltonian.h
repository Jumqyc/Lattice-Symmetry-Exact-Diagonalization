#pragma once

#include <complex>
#include <vector>

#include "base_types.h"
#include "hilbert.h"

namespace ed {

/**
 * @detail A term is one product of local operators, one operator per site, and
 *         an operator is stored by its non-zero entries only.  ``in`` and
 *         ``out`` are local indices, so a term maps a basis state to a sum of
 *         basis states without ever forming a matrix on the full space.
 * @short One local operator: out[j] <- in[j] with coefficient coeff[j].
 */
struct LocalTerm {
    std::vector<int> out;
    std::vector<int> in;
    std::vector<Complex> coeff;
};

/**
 * @detail One coupling is a set of sites plus the terms that act on them, which
 *         is exactly the Python ``Coupling``: several terms on the same sites
 *         are added, so a Heisenberg bond is three terms.
 * @short A product of local operators acting on a fixed set of sites.
 */
struct CouplingTerm {
    std::vector<int> sites;
    std::vector<LocalTerm> local;
};

/**
 * @param basis the computational basis.
 * @param term one term of the Hamiltonian.
 * @param word the packed state to act on.
 * @param digits the same state as local indices, one per site.
 * @param out appended with ``(image index, coefficient)`` for every image that
 *        lies inside the basis.
 * @detail The recursion walks the term's sites and carries the packed word, so
 *         no state vector is built; the caller unpacks the state once and passes
 *         the digits, which is what keeps a term application free of allocation.
 * @short Append the images of one term acting on one state.
 */
void apply_term(const Basis& basis, const CouplingTerm& term, std::uint64_t word,
                const std::vector<int>& digits, std::vector<std::pair<int, Complex>>& out);

/**
 * @param matrix a matrix in CSR form.
 * @detail The transposed matrix, also in CSR form; the sector assembly needs to
 *         walk columns as easily as rows.
 * @short Transpose of a sparse matrix.
 */
SparseMatrix transpose(const SparseMatrix& matrix);

/**
 * @param matrix a matrix in CSR form.
 * @param vector a dense vector, length ``matrix.cols``.
 * @detail Plain CSR mat-vec; the workhorse of the Lanczos solver.
 * @short ``matrix * vector``.
 */
std::vector<Complex> multiply(const SparseMatrix& matrix, const std::vector<Complex>& vector);

}  // namespace ed
