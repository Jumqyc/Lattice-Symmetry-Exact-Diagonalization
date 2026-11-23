#include "group.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ed {

namespace {

/**
 * @detail The identity is the element e with e * x = x for every x; the table is
 *         assumed to be the table of a group, so a single row check suffices.
 * @short Index of the identity element.
 */
int find_identity(const std::vector<std::vector<int>>& mult)
{
    const int n = static_cast<int>(mult.size());
    for (int e = 0; e < n; ++e) {
        bool identity = true;
        for (int x = 0; x < n && identity; ++x) identity = (mult[e][x] == x);
        if (identity) return e;
    }
    throw std::invalid_argument("the multiplication table has no identity");
}

/**
 * @detail inv[a] is the b with a * b = e; it exists because the table is a
 *         group table, and the search is O(n) per element.
 * @short Inverse of every element.
 */
std::vector<int> find_inverses(const std::vector<std::vector<int>>& mult, int identity)
{
    const int n = static_cast<int>(mult.size());
    std::vector<int> inv(n, -1);
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            if (mult[a][b] == identity) {
                inv[a] = b;
                break;
            }
    for (int a = 0; a < n; ++a)
        if (inv[a] < 0) throw std::invalid_argument("an element has no inverse");
    return inv;
}

/**
 * @detail The classes are the orbits of conjugation, x -> g x g^-1; walking the
 *         elements in order and marking the members keeps the labels stable.
 * @short Conjugacy classes and the class of every element.
 */
void find_classes(const std::vector<std::vector<int>>& mult,
                  const std::vector<int>& inv,
                  std::vector<std::vector<int>>& classes,
                  std::vector<int>& class_of)
{
    const int n = static_cast<int>(mult.size());
    std::vector<char> seen(n, 0);
    class_of.assign(n, -1);
    for (int a = 0; a < n; ++a) {
        if (seen[a]) continue;
        // the conjugates of a are g a g^-1
        std::vector<int> orbit;
        for (int g = 0; g < n; ++g) orbit.push_back(mult[mult[g][a]][inv[g]]);
        std::sort(orbit.begin(), orbit.end());
        orbit.erase(std::unique(orbit.begin(), orbit.end()), orbit.end());
        for (int x : orbit) {
            seen[x] = 1;
            class_of[x] = static_cast<int>(classes.size());
        }
        classes.push_back(orbit);
    }
}

/**
 * @param mult multiplication table.
 * @param classes conjugacy classes.
 * @param class_of class of every element.
 * @detail a[mu][nu][lam] counts the x in C_mu with x * y0 in C_lam for one fixed
 *         y0 in C_nu; the count does not depend on which y0 is used, because
 *         conjugating the pair does not change the class of the product.
 * @short Structure constants of the class algebra.
 */
std::vector<Eigen::MatrixXd> class_structure_constants(
    const std::vector<std::vector<int>>& mult,
    const std::vector<std::vector<int>>& classes,
    const std::vector<int>& class_of)
{
    const int m = static_cast<int>(classes.size());
    // N[mu] is the matrix with (N[mu])_{nu lam} = a^lam_{mu nu}
    std::vector<Eigen::MatrixXd> N(m, Eigen::MatrixXd::Zero(m, m));
    for (int mu = 0; mu < m; ++mu)
        for (int nu = 0; nu < m; ++nu) {
            const int y0 = classes[nu].front();
            for (int x : classes[mu]) N[mu](nu, class_of[mult[x][y0]]) += 1.0;
        }
    return N;
}

}  // namespace

CharacterTable character_table(const std::vector<std::vector<int>>& mult)
{
    const int n = static_cast<int>(mult.size());
    if (n == 0) throw std::invalid_argument("empty multiplication table");
    for (const auto& row : mult)
        if (static_cast<int>(row.size()) != n)
            throw std::invalid_argument("the multiplication table is not square");

    const int identity = find_identity(mult);
    const std::vector<int> inv = find_inverses(mult, identity);

    CharacterTable table;
    table.order = n;
    find_classes(mult, inv, table.classes, table.class_of);
    const int m = static_cast<int>(table.classes.size());
    const std::vector<Eigen::MatrixXd> N = class_structure_constants(mult, table.classes, table.class_of);

    // a generic combination of the commuting N[mu] has the central characters as
    // eigenvectors; the seed is fixed so that the result is reproducible
    std::mt19937 rng(20261123u);
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::MatrixXd combination = Eigen::MatrixXd::Zero(m, m);
    for (int mu = 0; mu < m; ++mu) combination += normal(rng) * N[mu];

    Eigen::ComplexEigenSolver<Eigen::MatrixXd> solver(combination);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("the class-algebra combination could not be diagonalised");

    std::vector<Eigen::VectorXcd> characters;   // one row of chi per irrep
    std::vector<double> dims;
    for (int rho = 0; rho < m; ++rho) {
        const Eigen::VectorXcd v = solver.eigenvectors().col(rho);
        const double norm_v = std::norm(v.norm());
        if (!(norm_v > 0.0) || !std::isfinite(norm_v)) continue;

        // omega(mu) is the eigenvalue of N[mu] on this common eigenvector
        std::vector<Complex> omega(m);
        double gram = 0.0;
        for (int mu = 0; mu < m; ++mu) {
            const Complex numerator = v.dot(N[mu].cast<Complex>() * v);
            omega[mu] = numerator / norm_v;
            gram += std::norm(omega[mu]) / static_cast<double>(table.classes[mu].size());
        }
        if (!(gram > 0.0)) continue;
        const double dim2 = static_cast<double>(n) / gram;
        if (!(dim2 > 1e-6)) continue;   // spurious eigenvector of a degenerate block
        const double dim = std::sqrt(dim2);

        Eigen::VectorXcd chi(m);
        for (int mu = 0; mu < m; ++mu)
            chi[mu] = dim * omega[mu] / static_cast<double>(table.classes[mu].size());
        characters.push_back(chi);
        dims.push_back(dim);
    }

    if (static_cast<int>(characters.size()) != m)
        throw std::runtime_error("the class algebra did not produce one irrep per class");

    // deterministic order: by dimension, then by the characters of the first
    // classes, so that repeated runs and different machines agree
    std::vector<int> order(characters.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (std::abs(dims[a] - dims[b]) > 1e-9) return dims[a] < dims[b];
        for (int mu = 0; mu < m; ++mu) {
            if (std::abs(characters[a][mu].real() - characters[b][mu].real()) > 1e-8)
                return characters[a][mu].real() < characters[b][mu].real();
            if (std::abs(characters[a][mu].imag() - characters[b][mu].imag()) > 1e-8)
                return characters[a][mu].imag() < characters[b][mu].imag();
        }
        return false;
    });

    table.chars.resize(static_cast<int>(order.size()), m);
    table.dims.resize(static_cast<int>(order.size()));
    for (int row = 0; row < static_cast<int>(order.size()); ++row) {
        table.chars.row(row) = characters[order[row]].transpose();
        table.dims[row] = dims[order[row]];
    }
    return table;
}

}  // namespace ed
