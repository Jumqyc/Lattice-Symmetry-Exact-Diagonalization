#pragma once

#include <functional>
#include <vector>

#include "base_types.h"

namespace ed {

/**
 * @detail One application of the operator being diagonalized.  The recurrence
 *         needs nothing else, so keeping the product behind a function is what
 *         lets the same solver work for a stored matrix and for an operator that
 *         is applied term by term -- which is where the memory of a five-by-five
 *         lattice stops being affordable.
 * @short A matrix-vector product.
 */
using MatrixVector = std::function<std::vector<Complex>(const std::vector<Complex>&)>;

/**
 * @param multiply the product, on vectors of length *dimension*.
 * @param dimension length of the vectors the product takes.
 * @param count how many eigenvalues, smallest first.
 * @param iterations the budget, in steps of the recurrence, over all cycles.
 * @param tolerance convergence threshold on the residual, relative to the
 *        magnitude of the value.
 * @param workers how many threads to use; 0 means one per hardware thread.
 * @detail Lanczos with full reorthogonalisation and a thick restart.
 *         Full reorthogonalisation is what keeps the recurrence trustworthy on
 *         the degenerate spectrum a symmetry sector always has, but it reads
 *         every stored vector at every step, so a step gets slower as the
 *         recurrence grows and the basis stops fitting in cache long before it
 *         stops fitting in memory.  The recurrence therefore runs in short
 *         cycles: at the end of one, the wanted Ritz vectors and a few extra are
 *         kept, the rest are dropped, and the run continues with the projected
 *         matrix in arrowhead form.  The residual of a kept vector is the same
 *         direction as that of every other one -- the next Lanczos vector scaled
 *         by the last component of its projected eigenvector -- so the residual
 *         test survives the restart unchanged.  The projected matrix is small and
 *         is diagonalized directly by Jacobi rotations, which also hands back the
 *         eigenvectors the restart needs.
 *         A single-vector recurrence resolves *distinct* eigenvalues only: an
 *         exactly degenerate level contributes one Ritz vector per copy only
 *         when the restart keeps room for one.
 * @short Lowest eigenvalues of a Hermitian operator, given its product.
 * @throws std::runtime_error if a requested Ritz value is not converged.
 */
std::vector<double> lowest_eigenvalues(const MatrixVector& multiply, int dimension,
                                       int count, int iterations = 200,
                                       double tolerance = 1e-10, int workers = 0);

/**
 * @param matrix a sparse Hermitian matrix.
 * @param count how many eigenvalues, smallest first.
 * @param iterations Arnoldi dimension.
 * @param tolerance convergence threshold on the residual.
 * @param workers how many threads to use; 0 means one per hardware thread.
 * @detail The same solver, with the product taken from a stored CSR matrix.
 * @short Lowest eigenvalues of a sparse Hermitian matrix.
 * @throws std::runtime_error if a requested Ritz value is not converged.
 */
std::vector<double> lowest_eigenvalues(const SparseMatrix& matrix, int count,
                                       int iterations = 200, double tolerance = 1e-10,
                                       int workers = 0);

}  // namespace ed
