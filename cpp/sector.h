#pragma once

#include <complex>
#include <vector>

#include "base_types.h"
#include "hamiltonian.h"
#include "hilbert.h"

namespace ed {

/**
 * @detail Inside one momentum sector a symmetry operation acts as a phased
 *         permutation of the adapted basis: it sends one column onto one column
 *         and multiplies by a phase.  That phase is the Bloch phase of the
 *         change of orbit representative, times whatever the local orbitals
 *         contribute, and it is part of the representation -- dropping it makes
 *         the projector wrong.
 * @short A phased permutation of the columns of a momentum sector.
 */
struct PhasedPermutation {
    std::vector<int> act;              // act[x]: column that column x is sent to
    std::vector<Complex> phase;        // coefficient of that column
};

/**
 * @param columns the momentum-adapted basis of one sector.
 * @param basis the computational basis.
 * @param action one symmetry operation on the computational basis.
 * @detail One state of every column is taken as a probe: its image lies in the
 *         column the probe is sent to, so comparing the coefficient of the probe
 *         in its own column with the coefficient of its image in the image
 *         column gives the phase without forming any matrix.
 * @short Action of one operation on a momentum sector.
 * @return act and phase, one entry per column.
 * @throws std::runtime_error if the image of a column is outside the sector.
 */
PhasedPermutation sector_action(const SparseColumns& columns, const Basis& basis,
                                const MonomialAction& action);

/**
 * @detail The isotypic component of one irrep is a direct sum over the orbits of
 *         the little group: the projector commutes with the group and the group
 *         permutes the columns inside an orbit, so no orbit mixes with another.
 *         Working per orbit keeps every linear algebra step of the size of the
 *         orbit (at most |G|) instead of the size of the sector (up to 2e5 at
 *         5x5), which is what makes a dense M x M projector unnecessary.
 * @short An orthonormal basis of one irrep's subspace, one block per orbit.
 */
struct SectorVectors {
    std::vector<int> start;            // size = #vectors + 1, into the entries
    std::vector<int> columns;          // sector column of every entry
    std::vector<Complex> data;         // coefficient of every entry
    std::vector<int> orbit_of_vector;  // orbit that each vector was built on

    int vectors() const { return static_cast<int>(start.size()) - 1; }
};

/**
 * @param actions the action of every element of the little group on the columns.
 * @param characters chi_alpha of the irrep, one per element of *actions*.
 * @param dim dimension d_alpha of the irrep.
 * @param column_count number of columns of the sector.
 * @detail For every orbit, the projector (d_alpha/|G|) sum_r chi*_alpha(r) U(r)
 *         is built on the orbit (orbit x orbit) and diagonalised; the
 *         eigenvectors with eigenvalue one are the adapted vectors.  The result
 *         is orthonormal by construction and every vector lives on one orbit.
 * @short Orthonormal basis of the alpha-isotypic subspace.
 */
SectorVectors isotypic_basis(const std::vector<PhasedPermutation>& actions,
                             const std::vector<Complex>& characters,
                             int dim, int column_count);

/**
 * @param hamiltonian the Hamiltonian in the computational basis.
 * @param columns the momentum-adapted basis.
 * @detail ``B^T H B``, assembled in CSR form without ever forming a dense
 *         matrix of the size of the sector: the product is taken column by
 *         column, so the cost is the number of non-zeros of ``H B`` times the
 *         number of non-zeros of the column of ``B``.
 * @short The Hamiltonian in the momentum-adapted basis.
 */
SparseMatrix sector_hamiltonian(const SparseMatrix& hamiltonian, const SparseColumns& columns);

/**
 * @param matrix the Hamiltonian of a sector.
 * @param vectors an orthonormal set of sector vectors.
 * @detail ``V^T H V``, returned dense because the block is what the
 *         eigensolver consumes.  The cost is the number of non-zeros of the
 *         sector Hamiltonian times the number of vectors.
 * @short The Hamiltonian block in the adapted basis.
 * @return Column-major dense block, ``vectors.vectors()`` square.
 */
std::vector<Complex> block_matrix(const SparseMatrix& matrix, const SectorVectors& vectors);

/**
 * @param block column-major dense Hermitian block.
 * @param size side of the block.
 * @detail Dense self-adjoint eigenvalues, ascending; the small-block path.
 * @short All eigenvalues of a dense Hermitian block.
 */
std::vector<double> eigenvalues(const std::vector<Complex>& block, int size);

/**
 * @param matrix a sparse Hermitian matrix.
 * @param count how many eigenvalues, smallest first.
 * @param iterations Arnoldi dimension; the three-term recurrence needs enough
 *        vectors to resolve a degenerate low-lying spectrum.
 * @param tolerance convergence threshold on the residual.
 * @detail Lanczos with full reorthogonalisation, which is what makes it
 *         stable for clustered spectra at the price of storing ``iterations``
 *         vectors.  A single-vector recursion resolves *distinct* eigenvalues
 *         only: an exactly degenerate level contributes one Ritz vector, not
 *         one per copy.  Multiplicities therefore have to be read elsewhere --
 *         from the dense solver for a block that fits, or from the character
 *         trace formula -- and the caller must not assume that the returned
 *         values are the k lowest counting multiplicity.
 * @short Lowest eigenvalues of a sparse Hermitian matrix.
 */
std::vector<double> lowest_eigenvalues(const SparseMatrix& matrix, int count,
                                       int iterations = 200, double tolerance = 1e-10);

}  // namespace ed
