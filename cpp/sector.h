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
                             int dim, int column_count, int threads = 0);

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
 * @param threads how many workers to use; 0 means one per hardware thread.
 * @detail ``B^T H B`` with the Hamiltonian applied term by term: ``H B`` is
 *         accumulated column by column from the images the terms produce, so the
 *         Hamiltonian never exists as a matrix.  The cost is
 *         ``#states * #terms`` per sector; the memory is the size of the basis,
 *         which is what makes a five-by-five lattice possible at all.
 * @short The Hamiltonian in the momentum-adapted basis.
 */
SparseMatrix sector_hamiltonian(const Basis& basis, const std::vector<CouplingTerm>& couplings,
                                const SparseColumns& columns, int threads = 0);

/**
 * @param matrix the Hamiltonian of a sector, in the sector basis.
 * @param vectors an orthonormal set of sector vectors.
 * @param threads how many workers to use; 0 means one per hardware thread.
 * @detail ``V^T H V`` as a sparse matrix: both factors are sparse, the product
 *         is accumulated column by column, and every vector of *vectors* lives
 *         on one little-group orbit, so the block is as sparse as the sector
 *         Hamiltonian restricted to those orbits.  Nothing of the size of the
 *         sector is allocated on the way.
 * @short The Hamiltonian block in the adapted basis, sparse.
 * @return A CSR matrix of side ``vectors.vectors()``.
 */
SparseMatrix block_matrix(const SparseMatrix& matrix, const SectorVectors& vectors,
                          int threads = 0);

}  // namespace ed

namespace ed {

/**
 * @detail The momentum sector seen as orbit labels: which translation orbit a
 *         state belongs to, and which translation carries the orbit's
 *         representative to it.  That is all the projected product needs -- the
 *         phase of a matrix element is a power of the character at the second
 *         index -- so neither the momentum basis nor the sector Hamiltonian has to
 *         exist as a matrix.
 * @short State -> (orbit, translation) for one momentum sector.
 */
struct SectorMap {
    std::vector<int> orbit;          // one entry per state
    std::vector<short> translation;  // and the translation that reaches it
    std::vector<int> representative; // the word of every orbit
    std::vector<int> start;          // the states of an orbit: start[o]..start[o + 1]
    std::vector<int> states;
    std::vector<int> stabiliser_start;   // and the translations that fix the orbit
    std::vector<short> stabiliser;
    int orbits = 0;
};

/**
 * @param basis the computational basis.
 * @param translations one action per translation, in character order.
 * @detail Walks the basis once, the same way the momentum basis does, so the
 *         orbits and their representatives match what momentum_basis would have
 *         produced -- the representative is the smallest word of its orbit, which
 *         is what fixes the phases.
 * @short Label the translation orbits of a momentum sector.
 */
SectorMap sector_map(const Basis& basis, const std::vector<MonomialAction>& translations);

/**
 * @param full the orbit labels, as sector_map builds them.
 * @param characters the character of the momentum at every translation.
 * @detail An orbit whose column vanishes for this momentum -- the character is
 *         nontrivial on the orbit's stabiliser -- is dropped and its states are
 *         labelled -1, so the product skips the images that fall in it.  This is
 *         the same selection the momentum basis makes when it discards a column of
 *         vanishing norm.
 * @short Restrict the orbit labels to one momentum.
 */
SectorMap sector_subset(const SectorMap& full, const std::vector<Complex>& characters);

/**
 * @param basis the computational basis.
 * @param couplings the terms of the Hamiltonian.
 * @param sector the orbit labels of the momentum sector.
 * @param characters the character of the momentum at every translation.
 * @param vector the sector vector, one entry per orbit.
 * @param threads how many workers to use; 0 means one per hardware thread.
 * @detail One row per orbit: the Hamiltonian is applied to the orbit's
 *         representative term by term, and each image contributes to the orbit it
 *         belongs to, weighted by the character at the translation that reaches
 *         it.  No basis matrix and no sector Hamiltonian is ever formed, which is
 *         what lets a lattice too large for either of them be handled.
 * @short ``H_k vector``, matrix-free, in the space of orbit labels.
 */
std::vector<Complex> sector_apply(const Basis& basis, const std::vector<CouplingTerm>& couplings,
                                  const SectorMap& sector, const std::vector<Complex>& characters,
                                  const std::vector<Complex>& vector, int threads = 0);

/**
 * @param basis the computational basis.
 * @param sector the orbit labels of a momentum sector.
 * @param action one band of the little group, as a monomial action on states.
 * @detail The action of a little-group element on the *orbits*: the label a state
 *         belongs to is taken to the label its image belongs to, with the phase the
 *         action gives it.  The phase is the same for every state of an orbit,
 *         because the action commutes with translations, so one entry per orbit is
 *         enough -- and with it the little-group projection needs no basis matrix.
 * @short One little-group element acting on the orbit labels.
 */
PhasedPermutation action_on_orbits(const Basis& basis, const SectorMap& sector,
                                   const MonomialAction& action);

/**
 * @param basis the computational basis.
 * @param couplings the terms of the Hamiltonian.
 * @param sector the orbit labels of the momentum sector.
 * @param characters the character of the momentum at every translation.
 * @param vectors the adapted vectors of one irreducible sector, as coefficients on
 *        the orbits.
 * @param vector the sector vector, one entry per adapted vector.
 * @param threads how many workers to use; 0 means one per hardware thread.
 * @detail ``V^T B^T H B V`` applied to a vector, with neither ``B V``, ``B^T H B``
 *         nor the block written down: the vector is expanded into the states of the
 *         adapted orbits, the Hamiltonian is applied term by term, and the result
 *         is projected back.  The two state-space vectors that the expansion needs
 *         are the only large objects.
 * @short The Hamiltonian of one sector, applied matrix-free.
 */
std::vector<Complex> sector_apply_block(const Basis& basis,
                                        const std::vector<CouplingTerm>& couplings,
                                        const SectorMap& sector,
                                        const std::vector<Complex>& characters,
                                        const SectorVectors& vectors,
                                        const std::vector<Complex>& vector, int threads = 0);

}  // namespace ed
