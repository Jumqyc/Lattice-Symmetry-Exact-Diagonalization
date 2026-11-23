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
 * @param iterations Arnoldi dimension; the three-term recurrence needs enough
 *        vectors to resolve a degenerate low-lying spectrum.
 * @param tolerance convergence threshold on the residual, relative to the
 *        magnitude of the value.
 * @detail Lanczos with full reorthogonalisation, which is what makes it stable
 *         for clustered spectra at the price of storing ``iterations`` vectors.
 *         The tridiagonal recurrence is kept as its two diagonals: the Ritz
 *         values come from bisection on the inertia count of ``T - x``, and the
 *         residual of a Ritz pair is the last component of its eigenvector times
 *         the first neglected off-diagonal, obtained by inverse iteration on the
 *         tridiagonal.  No matrix of the size of the recurrence is formed.  The
 *         residual is tested every twenty steps and the recurrence stops as soon
 *         as it has converged, because a step costs more than the one before it.
 *         A single-vector recursion resolves *distinct* eigenvalues only: an
 *         exactly degenerate level contributes one Ritz vector, not one per copy,
 *         so the caller must not assume that the returned values are the k lowest
 *         counting multiplicity.
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
