#pragma once

#include <complex>
#include <vector>

#include "base_types.h"
#include "hilbert.h"
#include "parallel.h"

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
 * @detail A term is *monomial* when the local operator at every site sends one
 *         digit to one digit -- a permutation of the local states, or a partial
 *         one, times a phase.  Then the application needs no search at all: the
 *         digit indexes the image and the coefficient, which is the same shape as
 *         the group actions' MonomialAction and covers every term of the usual
 *         lattice models, because ``XX``, ``YY`` and ``ZZ`` are separate terms
 *         and each of them is monomial.
 * @short A term prepared for application without searching.
 */
struct PreparedTerm {
    const CouplingTerm* term = nullptr;
    bool monomial = false;
    std::vector<int> site;                 // sites of the term, in the order of the term
    std::vector<int> shift;                // bit offset of each of them
    std::vector<std::vector<int>> image;   // [position][digit], -1 where the term vanishes
    std::vector<std::vector<Complex>> phase;  // the same
};

/**
 * @param basis the computational basis, for the packing.
 * @param term one term of the Hamiltonian.
 * @detail The local operator of every site is read once into a table indexed by
 *         the local digit.  A digit that appears twice in the ``in`` column means
 *         the term is not monomial, and the table is dropped: the term then goes
 *         through the general walk.
 * @short Read a term into the tables that make its application direct.
 */
PreparedTerm prepare_term(const Basis& basis, const CouplingTerm& term);

/**
 * @param basis the computational basis.
 * @param prepared the term, prepared.
 * @param word the packed state to act on.
 * @param digits the same state as local indices.
 * @param prefix the charge in front of every site of that state.
 * @param source_index the index of that state in the basis.
 * @param out appended with ``(image index, coefficient)``.
 * @detail Monomial terms are applied with one table lookup per site; anything else
 *         falls back to the general walk.  The index of the image comes from the
 *         bit vector when there is one, and otherwise from the difference the
 *         changed digits make to the source's index, which is a walk over the span
 *         the term covers.
 * @short Append the images of one prepared term.
 */
void apply_prepared(const Basis& basis, const PreparedTerm& prepared, std::uint64_t word,
                    const std::vector<int>& digits, const std::vector<int>& prefix,
                    int source_index, std::vector<std::pair<int, Complex>>& out);

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
 * @param workers the pool to spread the rows over, or null for one thread.
 * @detail Plain CSR mat-vec, the workhorse of the Lanczos solver; the rows are
 *         independent and each worker writes only its own slice of the result.
 * @short ``matrix * vector``.
 */
std::vector<Complex> multiply(const SparseMatrix& matrix, const std::vector<Complex>& vector,
                              Workers* workers = nullptr);

}  // namespace ed
