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
 * @detail The rank structure is one bit per code-space word plus a sample every
 *         512 of them, so this is about sixty megabytes: past it the index is
 *         counted from the suffix table instead of read.
 * @short The largest code space the bit vector is built for.
 */
constexpr std::uint64_t kBitsetLimit = 1ull << 29;

}  // namespace

Basis::Basis(int sites, int local_dim, std::vector<int> charge, int sztot, bool with_table)
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
    sztot_ = sztot;
    constrained_ = sztot >= 0;
    if (constrained_) prepare_counts();
    if (with_table) prepare_bitset();
}

void Basis::prepare_counts()
{
    // The index of a word is the number of valid words before it in the
    // enumeration order, and those words split by the first digit at which they
    // differ from it: what is needed is how many ways the remaining sites can be
    // filled for a given remaining charge, filled from the end of the lattice
    // towards its start.  For a fixed number of ones this is the binomial table of
    // the combinatorial number system; for a general local dimension and charge
    // assignment it is the multinomial one.
    int lowest = charge_[0];
    int highest = charge_[0];
    for (const int value : charge_) {
        lowest = std::min(lowest, value);
        highest = std::max(highest, value);
    }
    low_ = lowest * sites_;
    high_ = highest * sites_;
    width_ = high_ - low_ + 1;
    count_.assign(static_cast<std::size_t>(sites_ + 1) * width_, 0);
    count_[static_cast<std::size_t>(sites_) * width_ + (0 - low_)] = 1;
    for (int k = sites_ - 1; k >= 0; --k)
        for (int charge = low_; charge <= high_; ++charge) {
            long long total = 0;
            for (int state = 0; state < local_dim_; ++state) {
                const int rest = charge - charge_[state];
                if (rest < low_ || rest > high_) continue;
                total += count_[static_cast<std::size_t>(k + 1) * width_ + (rest - low_)];
            }
            count_[static_cast<std::size_t>(k) * width_ + (charge - low_)] = total;
        }
}

void Basis::prepare_bitset()
{
    // One bit per code space word plus a sample every 512 of them: at five by five
    // that is four megabytes and a quarter, which answers a lookup in two
    // cache-resident reads.  Past about sixty megabytes the structure stops being
    // the cheaper option and is not built at all -- the index is then counted.
    space_ = 1;
    for (int i = 0; i < sites_; ++i) {
        space_ <<= bits_;
        if (space_ > kBitsetLimit) return;
    }
    bitset_.assign(static_cast<std::size_t>(space_ + 63) / 64, 0);
    for (int i = 0; i < size(); ++i)
        bitset_[states_[i] >> 6] |= 1ull << (states_[i] & 63);
    rank_.assign(static_cast<std::size_t>(space_ >> rank_shift_) + 2, 0);
    const std::size_t words_per_block = static_cast<std::size_t>(1) << (rank_shift_ - 6);
    int total = 0;
    for (std::size_t block = 0; block < rank_.size(); ++block) {
        rank_[block] = total;
        const std::size_t begin = block * words_per_block;
        const std::size_t end = std::min(begin + words_per_block, bitset_.size());
        for (std::size_t word = begin; word < end; ++word)
            total += __builtin_popcountll(bitset_[word]);
    }
}

void Basis::prefix_charges(const std::vector<int>& digits, std::vector<int>& prefix) const
{
    prefix.resize(static_cast<std::size_t>(sites_));
    int running = 0;
    for (int k = 0; k < sites_; ++k) {
        prefix[k] = running;
        running += charge_[digits[k]];
    }
}

bool Basis::index_incremental(const std::vector<int>& digits, const std::vector<int>& prefix,
                              const int* sites, const int* changed, int count,
                              int source_index, int& image_index) const
{
    if (count == 0) {
        image_index = source_index;
        return true;
    }
    if (!constrained_) {
        // without a charge constraint the rank is the mixed-radix value, and two
        // words that differ in a few digits differ by a few radix positions
        int delta = 0;
        for (int i = 0; i < count; ++i) {
            int weight = 1;
            for (int k = sites[i] + 1; k < sites_; ++k) weight *= local_dim_;
            delta += (changed[i] - digits[sites[i]]) * weight;
        }
        image_index = source_index + delta;
        return true;
    }
    // the image is in the sector only if the charges still cancel
    int drift = 0;
    for (int i = 0; i < count; ++i) drift += charge_[changed[i]] - charge_[digits[sites[i]]];
    if (drift != 0) return false;

    int delta = 0;
    int offset = 0;             // charge change from the sites before the current one
    int next = 0;               // index of the next changed site
    for (int k = sites[0]; k <= sites[count - 1]; ++k) {
        const int before_digit = digits[k];
        const int after_digit = (next < count && sites[next] == k) ? changed[next] : before_digit;
        // the charge still to be placed after site k: the source's, and the
        // image's, which differs by the change made at the sites before this one
        const int remaining_source = sztot_ - prefix[k];
        const int remaining_image = remaining_source - offset;
        int before = 0;
        int after = 0;
        for (int state = 0; state < before_digit; ++state) {
            const int rest = remaining_source - charge_[state];
            if (rest >= low_ && rest <= high_)
                before += static_cast<int>(count_[static_cast<std::size_t>(k + 1) * width_
                                                   + (rest - low_)]);
        }
        for (int state = 0; state < after_digit; ++state) {
            const int rest = remaining_image - charge_[state];
            if (rest >= low_ && rest <= high_)
                after += static_cast<int>(count_[static_cast<std::size_t>(k + 1) * width_
                                                  + (rest - low_)]);
        }
        delta += after - before;
        if (after_digit != before_digit) {
            offset += charge_[after_digit] - charge_[before_digit];
            ++next;
        }
    }
    image_index = source_index + delta;
    return true;
}

int Basis::index(std::uint64_t word) const
{
    // the bit vector, when it was built, answers in two reads
    if (!bitset_.empty()) {
        if (word >= space_) return -1;
        if ((bitset_[word >> 6] & (1ull << (word & 63))) == 0) return -1;
        const std::size_t block = static_cast<std::size_t>(word) >> rank_shift_;
        int result = rank_[block];
        const std::uint64_t begin = static_cast<std::uint64_t>(block) << rank_shift_;
        for (std::uint64_t at = begin; at + 64 <= word; at += 64)
            result += __builtin_popcountll(bitset_[at >> 6]);
        result += __builtin_popcountll(bitset_[word >> 6] & ((1ull << (word & 63)) - 1ull));
        return result;
    }
    // without a charge constraint every word is in the basis and the enumeration
    // order is the plain mixed-radix one
    if (!constrained_) {
        int rank = 0;
        for (int k = 0; k < sites_; ++k) {
            const int digit = this->digit(word, k);
            if (digit >= local_dim_) return -1;
            rank = rank * local_dim_ + digit;
        }
        return rank;
    }
    int rank = 0;
    int remaining = sztot_;
    for (int k = 0; k < sites_; ++k) {
        const int digit = this->digit(word, k);
        if (digit >= local_dim_) return -1;
        for (int state = 0; state < digit; ++state) {
            const int rest = remaining - charge_[state];
            if (rest < low_ || rest > high_) continue;
            rank += static_cast<int>(count_[static_cast<std::size_t>(k + 1) * width_
                                                + (rest - low_)]);
        }
        remaining -= charge_[digit];
    }
    return remaining == 0 ? rank : -1;
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


}  // namespace ed
