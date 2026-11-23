#pragma once

#include <complex>
#include <vector>

#include <Eigen/Dense>

namespace ed {

using Complex = std::complex<double>;

/**
 * @detail A finite group is described only by its multiplication table plus
 *         the algebra that the table generates; the character table is
 *         derived, never tabulated by hand.  This is what lets the same code
 *         serve the named point groups, an arbitrary subgroup (a little
 *         group), and later a group that carries fractional translations.
 * @short Character table of a finite group, plus its conjugacy classes.
 */
struct CharacterTable {
    int order = 0;                          // |G|
    std::vector<std::vector<int>> classes;  // class index -> element indices
    std::vector<int> class_of;              // element index -> class index
    Eigen::VectorXd dims;                   // irrep dimension, one per irrep
    Eigen::MatrixXcd chars;                 // (n_irreps, n_classes) characters
};

/**
 * @param mult multiplication table: mult[a][b] is the index of the product a*b.
 * @detail Conjugacy classes are the orbits of x -> g x g^-1.  The class sums
 *         e_mu = sum_{x in C_mu} x satisfy e_mu e_nu = sum_lam a^lam_{mu nu} e_lam,
 *         and the matrices N_mu with (N_mu)_{nu lam} = a^lam_{mu nu} commute, so a
 *         generic linear combination of them has the central characters
 *         omega_rho(e_mu) = |C_mu| chi_rho(mu) / d_rho as its eigenvectors.
 *         Reading omega off one eigenvector gives
 *         d_rho^2 = |G| / sum_mu |omega_rho(mu)|^2 / |C_mu|
 *         and chi_rho(mu) = d_rho omega_rho(mu) / |C_mu|.
 * @short Character table of the group described by *mult*.
 * @return The table, with the irreps sorted deterministically.
 */
CharacterTable character_table(const std::vector<std::vector<int>>& mult);

}  // namespace ed
