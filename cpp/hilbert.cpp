#include "hilbert.h"

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
    if (space <= kDenseLimit) {
        dense_limit_ = static_cast<int>(space);
        dense_.assign(space, -1);
        for (int i = 0; i < size(); ++i) dense_[states_[i]] = i;
    } else {
        sparse_.reserve(states_.size());
        for (int i = 0; i < size(); ++i) sparse_[states_[i]] = i;
    }
}

int Basis::digit(std::uint64_t word, int site) const
{
    const int shift = bits_ * (sites_ - 1 - site);
    return static_cast<int>((word >> shift) & ((1ull << bits_) - 1ull));
}

int Basis::index(std::uint64_t word) const
{
    if (!dense_.empty()) {
        if (word >= static_cast<std::uint64_t>(dense_limit_)) return -1;
        return dense_[word];
    }
    const auto it = sparse_.find(word);
    return it == sparse_.end() ? -1 : it->second;
}

std::vector<int> Basis::unpack(std::uint64_t word) const
{
    std::vector<int> digits(sites_);
    for (int i = 0; i < sites_; ++i) digits[i] = digit(word, i);
    return digits;
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
                             const std::vector<Complex>& characters)
{
    const int n = basis.size();
    const int groups = static_cast<int>(translations.size());
    if (static_cast<int>(characters.size()) != groups)
        throw std::invalid_argument("one character per translation is required");

    SparseColumns out;
    out.indptr.push_back(0);
    std::vector<char> seen(n, 0);
    std::vector<std::uint64_t> images(groups);
    std::vector<Complex> weights(groups);

    for (int start = 0; start < n; ++start) {
        if (seen[start]) continue;
        // walk the orbit, remember the smallest word as the representative
        std::uint64_t rep = std::numeric_limits<std::uint64_t>::max();
        for (int g = 0; g < groups; ++g) {
            images[g] = apply(basis, translations[g], basis.state(start)).first;
            const int at = basis.index(images[g]);
            if (at < 0)
                throw std::runtime_error("a translation left the basis; the sector is inconsistent");
            seen[at] = 1;
            rep = std::min(rep, images[g]);
        }

        // project from the representative, so the column phase is canonical
        std::map<std::uint64_t, Complex> accumulated;
        for (int g = 0; g < groups; ++g) {
            const auto image = apply(basis, translations[g], rep).first;
            accumulated[image] += std::conj(characters[g]) / static_cast<double>(groups);
        }
        double norm2 = 0.0;
        for (const auto& entry : accumulated) norm2 += std::norm(entry.second);
        const double norm = std::sqrt(norm2);
        if (!(norm > 1e-12)) continue;      // momentum incompatible with the stabiliser

        out.reps.push_back(basis.index(rep));
        for (const auto& entry : accumulated) {
            out.indices.push_back(basis.index(entry.first));
            out.data.push_back(entry.second / norm);
        }
        out.indptr.push_back(static_cast<int>(out.indices.size()));
    }
    return out;
}

}  // namespace ed
