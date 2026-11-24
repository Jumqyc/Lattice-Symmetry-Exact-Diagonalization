#pragma once

#include <complex>
#include <cstdint>
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
     * @param with_table build the bit vector when it fits, so the index is read
     *        instead of counted; false forces the counting path, which is what
     *        the two representations are compared with
     * @detail The states are enumerated with a depth-first walk that prunes on
     *         the remaining charge, so the cost is the number of surviving
     *         states times the local dimension, not ``local_dim^sites``.
     * @short Enumerate the basis.
     */
    Basis(int sites, int local_dim, std::vector<int> charge, int sztot = -1,
          bool with_table = true);

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
     * @detail The basis is the set of words whose local charges sum to the
     *         requested total, so its enumeration order is a mixed-radix ordering
     *         of those words and the index is the number of valid words before it.
     *         That count splits by the first digit at which a word differs, which
     *         makes it a sum over the digits of a precomputed table of suffix
     *         completions -- the multinomial generalization of the binomial
     *         ranking of combinations.  Nothing over the code space is stored, so
     *         the lookup needs the same few kilobytes whether the basis has a
     *         thousand states or ten billion; it costs a pass over the digits
     *         instead of one memory access, which is the trade it makes.  Both
     *         are therefore built when the code space is small enough for the bit
     *         vector -- a few tens of megabytes -- and that path is taken: it is
     *         the faster of the two, and the counting table is what the sizes
     *         beyond it fall back on.
     */
    int index(std::uint64_t word) const;

    /**
     * @param digits the local digits of the state the index is incremented from.
     * @param prefix the charge in front of every site: ``prefix[k]`` is the sum of
     *        the charges of the sites before ``k``.
     * @param sites the sites whose digit differs, ascending.
     * @param changed the new digit at each of them.
     * @param count how many sites differ.
     * @param source_index the index of the state the digits belong to.
     * @param image_index overwritten with the index of the image.
     * @return Whether the image is in the sector.
     * @detail The rank of a word is a sum over its digits, and two words that
     *         differ at a few sites agree on every digit before the first of them
     *         and after the last.  Between them the digits are the same but the
     *         charge still to be placed is shifted by the change so far, which is
     *         what makes this a walk over the changed span rather than over the
     *         lattice: a bond costs two table reads, a diagonal term costs none.
     */
    bool index_incremental(const std::vector<int>& digits, const std::vector<int>& prefix,
                           const int* sites, const int* changed, int count,
                           int source_index, int& image_index) const;

    /** Whether the bit vector was built, in which case index() is the cheaper way. */
    bool has_table() const { return !bitset_.empty(); }

    /**
     * @param digits the local digits of one state.
     * @param prefix overwritten with the charge in front of every site.
     * @short Fill the prefix charges ``index_delta`` works from.
     */
    void prefix_charges(const std::vector<int>& digits, std::vector<int>& prefix) const;

    /** The charge of one local state. */
    int local_charge(int state) const { return charge_[state]; }

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
    /**
     * @detail Fills the table of suffix completions the index is built from: for
     *         every suffix start and every reachable remaining charge, how many
     *         ways the rest of the lattice can be filled.
     */
    void prepare_counts();

    /**
     * @detail Builds the bit vector and its rank samples, and leaves them empty
     *         when the code space is larger than they should be.
     */
    void prepare_bitset();

    int sites_ = 0;
    int local_dim_ = 1;
    int bits_ = 0;
    std::vector<int> charge_;
    std::vector<std::uint64_t> states_;
    // how many completions of the suffix k..sites-1 have a given total charge,
    // indexed by k * width_ + (charge - low_): the index of a word is a sum over
    // these counts, so no structure over the code space is needed
    std::vector<long long> count_;
    // the rank structure, present only while it fits: one bit per code-space word
    // and one sample per 512 of them
    std::vector<std::uint64_t> bitset_;
    std::vector<int> rank_;
    int rank_shift_ = 9;
    std::uint64_t space_ = 0;
    int low_ = 0;
    int high_ = 0;
    int width_ = 0;
    bool constrained_ = false;
    int sztot_ = -1;
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
 * @param basis the computational basis, for the packing.
 * @param action the operation to test.
 * @return The number of sites the action shifts the packed word by, or -1 when it
 *         is not a pure rotation.
 * @detail A translation of a torus permutes the sites cyclically and carries no
 *         local phase, so its image is the packed word rotated by a whole number of
 *         fields -- a shift and an or, whatever the local dimension, because the
 *         fields move whole and no digit is reinterpreted.  Recognising that here
 *         is what lets the orbit walk cost a shift instead of a pass over the
 *         lattice.
 * @short The shift of a pure rotation, or -1.
 */
int rotation_shift(const Basis& basis, const MonomialAction& action);

/**
 * @param word the packed state.
 * @param shift the number of bits to rotate it by, left.
 * @param total how many bits the word carries.
 * @short Rotate a packed word cyclically.
 */
std::uint64_t rotate_word(std::uint64_t word, int shift, int total);

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



}  // namespace ed
