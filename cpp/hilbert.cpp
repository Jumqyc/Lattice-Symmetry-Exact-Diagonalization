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
    sztot_ = sztot;
    constrained_ = sztot >= 0;
    if (constrained_) prepare_counts();
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

int Basis::index(std::uint64_t word) const
{
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
