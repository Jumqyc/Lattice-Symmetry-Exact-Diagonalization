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
 * @param couplings the terms of the Hamiltonian.
 * @detail Every state is walked once and every term is applied to it, so the
 *         cost is ``#states * #terms`` and the result is assembled in CSR form
 *         directly.
 * @short The Hamiltonian in the computational basis.
 */
SparseMatrix assemble(const Basis& basis, const std::vector<CouplingTerm>& couplings);

/**
 * @param matrix the Hamiltonian to average over the group.
 * @param basis the computational basis.
 * @param actions one action per group element, translations included.
 * @detail The result is ``(1/|G|) sum_g U(g) H U(g)^-1``, i.e. the projector
 *         onto the operators that commute with the group.  The invariant case --
 *         every coupling already respecting the symmetry, which is the usual one
 *         -- is detected first and returns the input untouched, because
 *         averaging a matrix that is already symmetric is a waste of a factor
 *         of |G| and, more importantly, an unchanged H is the honest outcome:
 *         silently averaging a non-invariant Hamiltonian would change the
 *         physics without saying so.
 * @short Symmetric part of the Hamiltonian.
 */
SparseMatrix symmetrise(const SparseMatrix& matrix, const Basis& basis,
                        const std::vector<MonomialAction>& actions);

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
