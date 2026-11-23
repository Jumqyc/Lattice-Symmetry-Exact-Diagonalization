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

    int vectors() const { return static_cast<int>(start.size()) - 1; }
};

/**
 * @param actions the action of every element of the little group on the columns.
 * @param characters chi_alpha of the irrep, one per element of *actions*.
 * @param dim dimension d_alpha of the irrep.
 * @param column_count number of columns of the sector.
 * @detail For every orbit the projected basis vectors (d_alpha/|G|) sum_r
 *         chi*_alpha(r) U(r) e_x are formed and orthonormalized in place; the
 *         image of the projector is spanned by them, so this is an orthonormal
 *         basis of the alpha-isotypic component of the orbit.  The work is done
 *         on the size of the orbit, and the dimension that comes out is
 *         compared against the trace of the projector, which the character
 *         fixes: a mismatch means the action and the character table disagree.
 * @short Orthonormal basis of the alpha-isotypic subspace.
 * @throws std::runtime_error if the orbit's rank is not the one the character
 *         predicts.
 */
SectorVectors isotypic_basis(const std::vector<PhasedPermutation>& actions,
                             const std::vector<Complex>& characters,
                             int dim, int column_count);

/**
 * @param column_count number of columns of the sector.
 * @detail The adapted set is the sector itself: one vector per column, with
 *         unit weight.  It is what the isotypic basis degenerates to when the
 *         little group is trivial, and it keeps the block assembly on the same
 *         path as every other sector instead of special-casing it.
 * @short The sector basis itself, as a SectorVectors.
 */
SectorVectors identity_vectors(int column_count);

/**
 * @param basis the computational basis.
 * @param couplings the terms of the Hamiltonian.
 * @param columns the momentum-adapted basis.
 * @detail ``B^T H B`` with the Hamiltonian applied term by term: ``H B`` is
 *         accumulated column by column from the images the terms produce, so the
 *         Hamiltonian never exists as a matrix.  The cost is
 *         ``#states * #terms`` per sector; the memory is the size of the basis,
 *         which is what makes a five-by-five lattice possible at all.
 * @short The Hamiltonian in the momentum-adapted basis.
 */
SparseMatrix sector_hamiltonian(const Basis& basis, const std::vector<CouplingTerm>& couplings,
                                const SparseColumns& columns);

/**
 * @param matrix the Hamiltonian of a sector, in the sector basis.
 * @param vectors an orthonormal set of sector vectors.
 * @detail ``V^T H V`` as a sparse matrix: both factors are sparse, the product
 *         is accumulated column by column, and every vector of *vectors* lives
 *         on one little-group orbit, so the block is as sparse as the sector
 *         Hamiltonian restricted to those orbits.  Nothing of the size of the
 *         sector is allocated on the way.
 * @short The Hamiltonian block in the adapted basis, sparse.
 * @return A CSR matrix of side ``vectors.vectors()``.
 */
SparseMatrix block_matrix(const SparseMatrix& matrix, const SectorVectors& vectors);

}  // namespace ed
