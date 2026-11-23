#pragma once

#include <complex>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "base_types.h"

namespace ed {

/**
 * @detail A basis state is a word of packed local indices, one field per site
 *         with ``bits = ceil(log2(local_dim))`` bits, site 0 in the most
 *         significant field.  That order is the lexicographic order of the
 *         digit tuples, which is what the Python reference implementation
 *         enumerates, so the two can be compared index by index.
 *         Packing is not an optimisation for its own sake: at 5x5 with Sz=0
 *         there are 5.2e6 states, and one tuple per state (what Python stores)
 *         does not fit in memory.
 * @short The computational basis: enumeration and the state -> index map.
 */
class Basis {
public:
    /**
     * @param sites number of sites.
     * @param local_dim dimension of the local space (the same for every site).
     * @param charge charge of every local state, used to select the sector.
     * @param sztot if not negative, keep only states whose charges sum to it.
     * @detail The states are enumerated with a depth-first walk that prunes on
     *         the remaining charge, so the cost is the number of surviving
     *         states times the local dimension, not ``local_dim^sites``.
     * @short Enumerate the basis.
     */
    Basis(int sites, int local_dim, std::vector<int> charge, int sztot = -1);

    int size() const { return static_cast<int>(states_.size()); }
    int sites() const { return sites_; }
    int local_dim() const { return local_dim_; }
    int bits() const { return bits_; }
    std::uint64_t state(int index) const { return states_[index]; }

    /**
     * @param word packed basis state.
     * @detail Field ``i`` of the word is the local index at site ``i``.
     * @short Local index at one site.
     */
    int digit(std::uint64_t word, int site) const;

    /**
     * @param word packed basis state.
     * @return The index of *word* in the basis, or -1 if it is not in it.
     * @detail Three representations, chosen by the size of the code space: a
     *         direct table when it is small, a bit vector with a rank sample every
     *         512 states when it is large, and a hash map when even the bit
     *         vector would be huge.  The bit vector matters: a table of one int
     *         per code-space word is tens of megabytes at five by five and every
     *         lookup in it is a cache miss, while the bit vector and its samples
     *         are a few megabytes and stay resident.
     */
    int index(std::uint64_t word) const;

    /**
     * @param word packed basis state.
     * @short The state as digit tuples, for comparisons and debugging.
     */
    std::vector<int> unpack(std::uint64_t word) const;

    /**
     * @param word packed basis state.
     * @param digits overwritten with the local index of every site.
     * @short The same, without allocating: the caller keeps the buffer.
     */
    void unpack_into(std::uint64_t word, std::vector<int>& digits) const;

private:
    void build_lookup();

    int sites_ = 0;
    int local_dim_ = 1;
    int bits_ = 0;
    std::vector<int> charge_;
    std::vector<std::uint64_t> states_;
    // one bit per code-space word when that is cheaper than a table, and a hash
    // map when even the bits would not fit
    std::vector<int> dense_;
    int dense_limit_ = 0;
    std::vector<std::uint64_t> bitset_;
    std::vector<int> rank_;                 // one entry per 512 words of code space
    int rank_shift_ = 9;
    std::unordered_map<std::uint64_t, int> sparse_;
};

/**
 * @detail One symmetry operation on the computational basis, in the monomial
 *         form the framework requires: it moves every site (``site_perm``, in
 *         the preimage convention) and multiplies the local index at every
 *         site by a permutation and a phase.  A pure site permutation has the
 *         local tables equal to the identity.
 * @short A monomial action: site permutation plus local permutation and phase.
 */
struct MonomialAction {
    std::vector<int> site_perm;              // length = sites, preimage map
    std::vector<int> local_perm;             // length = sites * local_dim
    std::vector<Complex> local_phase;        // length = sites * local_dim
};

/**
 * @param basis the computational basis.
 * @param action the operation.
 * @param word a basis state.
 * @detail The image is assembled field by field: the value at site ``i`` comes
 *         from site ``site_perm[i]`` and is then permuted by the local table of
 *         site ``i``.  This is exactly the Python ``MonomialAction.on_state``,
 *         including the phase, so the two implementations can be compared.
 * @short ``U(g)|word> = phase * |image>``.
 */
std::pair<std::uint64_t, Complex> apply(const Basis& basis, const MonomialAction& action,
                                        std::uint64_t word);

/**
 * @detail Columns are the momentum-adapted vectors
 *         ``|k;x> = P_k|x> / ||P_k|x>||`` with ``P_k = sum_t chi_k(t)^* t / |G_tr|``,
 *         one column per translation-orbit representative; rows are basis
 *         indices.  Columns are stored compressed by column, rows sorted.
 * @short A set of sparse columns over the computational basis.
 */
struct SparseColumns {
    int column_count = 0;           // how many columns the sector has
    std::vector<int> indptr;        // size = columns + 1
    std::vector<int> indices;       // basis index of each entry
    std::vector<Complex> data;      // coefficient of each entry

    int columns() const { return column_count; }
};

/**
 * @param basis the computational basis.
 * @param translations the site action of each translation, in preimage form.
 * @param characters chi_k(t) of each translation, for the momentum *k*.
 * @detail Orbits are walked once; the representative is the smallest packed
 *         state of the orbit, and the projection is taken *from the
 *         representative*, not from whichever state started the walk, so the
 *         global phase of every column matches the Python implementation.
 *         Orbits whose projected norm vanishes (the momentum is incompatible
 *         with the translation stabiliser of the state) are dropped.
 * @short The momentum-adapted orthonormal basis of one momentum sector.
 */
SparseColumns momentum_basis(const Basis& basis,
                             const std::vector<MonomialAction>& translations,
                             const std::vector<Complex>& characters, int threads = 0);

}  // namespace ed
