#include "hilbert.h"

#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>

namespace ed {

namespace {

/**
 * @detail Reserves the direct index table when the whole code space is small
 *         enough to index by the packed word; otherwise a hash map is used and
 *         only the states that survived the charge filter appear in it.
 * @short How many words the direct table can address.
 */
constexpr int kDenseLimit = 1 << 28;   // 2^28 words, about 1 GB of int

}  // namespace

Basis::Basis(int sites, int local_dim, std::vector<int> charge, int sztot)
    : sites_(sites), local_dim_(local_dim), charge_(std::move(charge))
{
    if (sites_ <= 0) throw std::invalid_argument("the lattice has no sites");
    if (local_dim_ <= 0) throw std::invalid_argument("the local dimension must be positive");
    if (static_cast<int>(charge_.size()) != local_dim_)
        throw std::invalid_argument("one charge per local state is required");
    bits_ = 0;
    while ((1 << bits_) < local_dim_) ++bits_;
    if (static_cast<long>(bits_) * sites_ > 64)
        throw std::invalid_argument(
            "a packed state does not fit in 64 bits for this lattice; the packing has to be "
            "extended to several words first");
    if (sztot >= 0) {
        int total = 0;
        for (int i = 0; i < sites_; ++i) {
            int best = charge_[0];
            for (int v = 1; v < local_dim_; ++v) best = std::max(best, charge_[v]);
            total += best;
        }
        if (sztot > total)
            throw std::invalid_argument("the requested charge cannot be reached");
    }

    // depth-first enumeration with pruning on the remaining charge; site 0 is
    // filled first so the order matches the lexicographic digit order
    const std::uint64_t field = static_cast<std::uint64_t>(bits_) * (sites_ - 1);
    std::vector<int> digits(sites_, 0);
    std::function<void(int, std::uint64_t, int)> walk =
        [&](int site, std::uint64_t word, int remaining) {
            if (site == sites_) {
                if (sztot < 0 || remaining == 0) states_.push_back(word);
                return;
            }
            for (int v = 0; v < local_dim_; ++v) {
                const int left = remaining - charge_[v];
                if (sztot >= 0 && (left < 0)) continue;
                walk(site + 1, word | (static_cast<std::uint64_t>(v) << (field - static_cast<std::uint64_t>(bits_) * site)),
                     sztot >= 0 ? left : 0);
            }
        };
    walk(0, 0ull, sztot >= 0 ? sztot : 0);
    build_lookup();
}

void Basis::build_lookup()
{
    const long long space = [&] {
        long long s = 1;
        for (int i = 0; i < sites_; ++i) {
            s <<= bits_;
            if (s > (1LL << 40)) break;
        }
        return s;
    }();
    // A table of one int per word is the fastest lookup and is used while it is
    // small; past that one bit per word plus a rank sample every 512 words is
    // thirty times smaller and stays in cache, which is what a sector product
    // cares about; past a bit vector of 128 MB even that is dropped for a hash.
    if (space <= (1LL << 16)) {
        dense_limit_ = static_cast<int>(space);
        dense_.assign(static_cast<std::size_t>(space), -1);
        for (int i = 0; i < size(); ++i) dense_[states_[i]] = i;
        return;
    }
    if (space <= (1LL << 30)) {
        dense_limit_ = static_cast<int>(space);
        bitset_.assign(static_cast<std::size_t>(space + 63) / 64, 0);
        for (int i = 0; i < size(); ++i)
            bitset_[states_[i] >> 6] |= 1ull << (states_[i] & 63);
        rank_.assign(static_cast<std::size_t>(space >> rank_shift_) + 2, 0);
        const std::size_t words_per_block = static_cast<std::size_t>(1) << (rank_shift_ - 6);
        int total = 0;
        for (std::size_t block = 0; block < rank_.size(); ++block) {
            rank_[block] = total;
            const std::size_t begin = block * words_per_block;
            const std::size_t end = std::min(begin + words_per_block, bitset_.size());
            for (std::size_t word = begin; word < end; ++word)
                total += __builtin_popcountll(bitset_[word]);
        }
        return;
    }
    sparse_.reserve(states_.size());
    for (int i = 0; i < size(); ++i) sparse_[states_[i]] = i;
}

int Basis::index(std::uint64_t word) const
{
    if (!dense_.empty()) {
        if (word >= static_cast<std::uint64_t>(dense_limit_)) return -1;
        return dense_[word];
    }
    if (!bitset_.empty()) {
        if (word >= static_cast<std::uint64_t>(dense_limit_)) return -1;
        if ((bitset_[word >> 6] & (1ull << (word & 63))) == 0) return -1;
        // the rank: the sample of this block, then the words before the word
        // itself, then the bits below the state inside it
        const std::size_t block = static_cast<std::size_t>(word) >> rank_shift_;
        int result = rank_[block];
        const std::uint64_t begin = static_cast<std::uint64_t>(block) << rank_shift_;
        for (std::uint64_t at = begin; at + 64 <= word; at += 64)
            result += __builtin_popcountll(bitset_[at >> 6]);
        result += __builtin_popcountll(bitset_[word >> 6] & ((1ull << (word & 63)) - 1ull));
        return result;
    }
    const auto it = sparse_.find(word);
    return it == sparse_.end() ? -1 : it->second;
}

int Basis::digit(std::uint64_t word, int site) const
{
    const int shift = bits_ * (sites_ - 1 - site);
    return static_cast<int>((word >> shift) & ((1ull << bits_) - 1ull));
}


std::vector<int> Basis::unpack(std::uint64_t word) const
{
    std::vector<int> digits;
    unpack_into(word, digits);
    return digits;
}

void Basis::unpack_into(std::uint64_t word, std::vector<int>& digits) const
{
    digits.resize(static_cast<std::size_t>(sites_));
    for (int i = 0; i < sites_; ++i) digits[i] = digit(word, i);
}

std::pair<std::uint64_t, Complex> apply(const Basis& basis, const MonomialAction& action,
                                        std::uint64_t word)
{
    const int sites = basis.sites();
    const int dim = basis.local_dim();
    if (static_cast<int>(action.site_perm.size()) != sites)
        throw std::invalid_argument("the site permutation does not match the lattice");
    if (static_cast<int>(action.local_perm.size()) != sites * dim)
        throw std::invalid_argument("the local permutation table has the wrong size");

    const int bits = basis.bits();
    std::uint64_t image = 0;
    Complex phase{1.0, 0.0};
    for (int i = 0; i < sites; ++i) {
        const int source = action.site_perm[i];
        const int value = basis.digit(word, source);
        const int moved = action.local_perm[i * dim + value];
        image |= static_cast<std::uint64_t>(moved) << (bits * (sites - 1 - i));
        phase *= action.local_phase[i * dim + value];
    }
    return {image, phase};
}

SparseColumns momentum_basis(const Basis& basis,
                             const std::vector<MonomialAction>& translations,
                             const std::vector<Complex>& characters, int threads)
{
    const int n = basis.size();
    const int groups = static_cast<int>(translations.size());
    if (static_cast<int>(characters.size()) != groups)
        throw std::invalid_argument("one character per translation is required");

    // The orbits are found by one walk over the basis: every state is visited once
    // and its orbit is marked, so the walk costs one application per state and the
    // representative of a column is the smallest word the translations reach from
    // it, which is what fixes the column's global phase.  The walk is serial
    // because it is a scan; it is also only a twenty-fifth of the work, since the
    // projection below runs once per orbit rather than once per state.
    std::vector<char> seen(static_cast<std::size_t>(n), 0);
    std::vector<std::uint64_t> representatives;
    representatives.reserve(static_cast<std::size_t>(n) / std::max(1, groups) + 1);
    for (int state = 0; state < n; ++state) {
        if (seen[state]) continue;
        std::uint64_t smallest = std::numeric_limits<std::uint64_t>::max();
        for (int g = 0; g < groups; ++g) {
            const std::uint64_t image = apply(basis, translations[g], basis.state(state)).first;
            const int at = basis.index(image);
            if (at < 0)
                throw std::runtime_error("a translation left the basis; the sector is inconsistent");
            seen[at] = 1;
            smallest = std::min(smallest, image);
        }
        representatives.push_back(smallest);
    }

    // the projection is one orbit at a time and the orbits are independent, so it
    // is spread over the workers; the columns come out in the order of the
    // representatives, which is the order the walk found them in
    std::vector<std::vector<std::pair<int, Complex>>> produced(representatives.size());
    parallel_ranges(static_cast<int>(representatives.size()), [&](int begin, int end) {
        std::map<std::uint64_t, Complex> accumulated;
        for (int index = begin; index < end; ++index) {
            accumulated.clear();
            for (int g = 0; g < groups; ++g) {
                const auto image = apply(basis, translations[g], representatives[index]).first;
                accumulated[image] += std::conj(characters[g]) / static_cast<double>(groups);
            }
            double norm2 = 0.0;
            for (const auto& entry : accumulated) norm2 += std::norm(entry.second);
            const double length = std::sqrt(norm2);
            if (!(length > 1e-12)) continue;    // momentum incompatible with the stabiliser
            std::vector<std::pair<int, Complex>>& column = produced[index];
            column.reserve(accumulated.size());
            for (const auto& entry : accumulated)
                column.emplace_back(basis.index(entry.first), entry.second / length);
        }
    }, threads);

    SparseColumns out;
    out.indptr.push_back(0);
    for (const auto& column : produced) {
        if (column.empty()) continue;
        ++out.column_count;
        for (const auto& entry : column) {
            out.indices.push_back(entry.first);
            out.data.push_back(entry.second);
        }
        out.indptr.push_back(static_cast<int>(out.indices.size()));
    }
    return out;
}

}  // namespace ed
