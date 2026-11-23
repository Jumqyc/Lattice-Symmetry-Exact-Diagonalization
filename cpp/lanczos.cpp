#include "lanczos.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "hamiltonian.h"

namespace ed {

namespace {

/**
 * @param alpha diagonal of the tridiagonal Lanczos matrix.
 * @param beta subdiagonal; ``beta[i]`` joins step ``i`` and step ``i+1``.
 * @param steps how many leading entries of the two diagonals form the matrix.
 * @param x the shift.
 * @detail The number of negative pivots of the LDL^T factor of ``T - x`` equals
 *         the number of eigenvalues below ``x`` (Sylvester's law of inertia).
 *         Only the two diagonals are touched, so no matrix is formed.
 * @short How many eigenvalues of the tridiagonal lie below ``x``.
 */
int count_below(const std::vector<double>& alpha, const std::vector<double>& beta,
                int steps, double x)
{
    int below = 0;
    double pivot = alpha[0] - x;
    if (pivot < 0.0) ++below;
    for (int i = 1; i < steps; ++i) {
        // a vanishing pivot is a shift sitting on an eigenvalue; nudging it off
        // keeps the inertia count, which is what the bisection uses
        if (std::abs(pivot) < 1e-300) pivot = 1e-300;
        pivot = (alpha[i] - x) - beta[i - 1] * beta[i - 1] / pivot;
        if (pivot < 0.0) ++below;
    }
    return below;
}

/**
 * @param alpha diagonal of the tridiagonal matrix.
 * @param beta subdiagonal.
 * @param steps how many leading entries form the matrix.
 * @param theta an eigenvalue of the tridiagonal.
 * @detail The last component of the eigenvector is what the Ritz residual needs.
 *         It comes from a few steps of inverse iteration with the LDL^T
 *         factorization of ``T - theta``, solved by forward and back
 *         substitution: both are linear in the number of steps, and the
 *         tridiagonal is never stored as a matrix.  The shift keeps the
 *         factorization away from the eigenvalue the iteration is aiming at.
 * @short Last component of the normalized eigenvector of the tridiagonal.
 */
double last_component(const std::vector<double>& alpha, const std::vector<double>& beta,
                      int steps, double theta)
{
    if (steps == 1) return 1.0;
    const double shift = 1e-12 * (1.0 + std::abs(theta));
    std::vector<double> diagonal(static_cast<std::size_t>(steps));
    std::vector<double> forward(static_cast<std::size_t>(steps));
    std::vector<double> vector(static_cast<std::size_t>(steps), 0.0);
    std::vector<double> previous(static_cast<std::size_t>(steps), 0.0);
    previous[0] = 1.0;
    for (int attempt = 0; attempt < 3; ++attempt) {
        diagonal[0] = alpha[0] - theta + shift;
        for (int i = 1; i < steps; ++i) {
            const double pivot = std::abs(diagonal[i - 1]) > 1e-14 ? diagonal[i - 1] : 1e-14;
            diagonal[i] = alpha[i] - theta + shift - beta[i - 1] * beta[i - 1] / pivot;
        }
        forward[0] = previous[0];
        for (int i = 1; i < steps; ++i) {
            const double pivot = std::abs(diagonal[i - 1]) > 1e-14 ? diagonal[i - 1] : 1e-14;
            forward[i] = previous[i] - beta[i - 1] * forward[i - 1] / pivot;
        }
        vector[steps - 1] = forward[steps - 1] / diagonal[steps - 1];
        for (int i = steps - 2; i >= 0; --i) {
            const double pivot = std::abs(diagonal[i]) > 1e-14 ? diagonal[i] : 1e-14;
            vector[i] = (forward[i] - beta[i] * vector[i + 1]) / pivot;
        }
        double norm = 0.0;
        for (const double value : vector) norm += value * value;
        norm = std::sqrt(norm);
        if (norm <= 0.0) return 0.0;
        for (int i = 0; i < steps; ++i) vector[i] /= norm;
        previous = vector;
    }
    return std::abs(vector[steps - 1]);
}

/**
 * @param alpha diagonal of the tridiagonal, length at least *steps*.
 * @param beta subdiagonal, length at least *steps* - 1.
 * @param steps length of the tridiagonal.
 * @param count how many of the lowest eigenvalues are wanted.
 * @detail Bisection on the inertia count: the tridiagonal is never stored, let
 *         alone diagonalized as a dense matrix.
 * @short The lowest eigenvalues of the tridiagonal, ascending.
 */
std::vector<double> ritz_values(const std::vector<double>& alpha,
                                const std::vector<double>& beta, int steps, int count)
{
    // Spectral bounds from the Gershgorin discs.  Every row has two off-diagonal
    // entries except the first and the last, so the bound is the row sum and not
    // a single beta: a bracket that cuts off the two lowest levels makes the
    // bisection converge to the edge of the bracket and report a value that is
    // not an eigenvalue at all.
    double low = alpha[0] - (steps > 1 ? std::abs(beta[0]) : 0.0);
    double high = alpha[steps - 1] + (steps > 1 ? std::abs(beta[steps - 2]) : 0.0);
    for (int i = 0; i < steps; ++i) {
        const double left = i > 0 ? std::abs(beta[i - 1]) : 0.0;
        const double right = i + 1 < steps ? std::abs(beta[i]) : 0.0;
        low = std::min(low, alpha[i] - left - right);
        high = std::max(high, alpha[i] + left + right);
    }
    const double resolution = (high - low) * 1e-15;
    std::vector<double> out(static_cast<std::size_t>(count), 0.0);
    for (int j = 0; j < count; ++j) {
        double left = low;
        double right = high;
        for (int iteration = 0; iteration < 200 && (right - left) > resolution; ++iteration) {
            const double middle = 0.5 * (left + right);
            if (count_below(alpha, beta, steps, middle) > j) right = middle;
            else left = middle;
        }
        out[j] = 0.5 * (left + right);
    }
    return out;
}

/**
 * @param alpha diagonal of the tridiagonal.
 * @param beta subdiagonal.
 * @param steps length of the tridiagonal.
 * @param beta_end the off-diagonal that would connect the last step to the next.
 * @param values the Ritz values to test.
 * @param tolerance residual threshold, relative to the magnitude of the value.
 * @detail The residual of a Ritz pair is the first neglected off-diagonal times
 *         the last component of its tridiagonal eigenvector.
 * @short Whether every requested Ritz value is converged.
 */
bool converged(const std::vector<double>& alpha, const std::vector<double>& beta,
               int steps, double beta_end, const std::vector<double>& values,
               double tolerance)
{
    for (const double value : values) {
        const double residual = beta_end * last_component(alpha, beta, steps, value);
        if (residual > tolerance * std::max(1.0, std::abs(value))) return false;
    }
    return true;
}

}  // namespace

std::vector<double> lowest_eigenvalues(const MatrixVector& multiply, int dimension,
                                       int count, int iterations, double tolerance)
{
    const int n = dimension;
    if (n <= 0) throw std::invalid_argument("the operator has no dimension");
    if (count <= 0 || count > n) throw std::invalid_argument("impossible number of eigenvalues");
    // with as many steps as the dimension the recurrence spans the whole space
    // and the Ritz values are exact; fewer steps give the usual approximation
    const int steps_max = std::min(iterations, n);
    // every step costs more than the one before it, because the recurrence is
    // reorthogonalized against all the vectors it already has, so the residual
    // is tested as the run goes and the recurrence stops as soon as it is
    // converged
    const int check_every = 20;

    // deterministic pseudo-random start: a constant vector can be orthogonal to
    // a whole degenerate multiplet, which would hide it from the recurrence
    std::mt19937 rng(20261123u);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<Complex> current(static_cast<std::size_t>(n));
    for (auto& value : current) value = Complex{normal(rng), normal(rng)};
    double norm = 0.0;
    for (const auto& value : current) norm += std::norm(value);
    norm = std::sqrt(norm);
    for (auto& value : current) value /= norm;

    // the recurrence is kept as its two diagonals: no matrix of the size of the
    // number of steps is ever formed
    std::vector<std::vector<Complex>> basis;
    basis.reserve(steps_max);
    std::vector<double> alpha(static_cast<std::size_t>(steps_max), 0.0);
    std::vector<double> beta(static_cast<std::size_t>(steps_max > 1 ? steps_max - 1 : 0), 0.0);
    std::vector<Complex> previous(static_cast<std::size_t>(n), Complex{0.0, 0.0});
    double beta_previous = 0.0;
    double beta_end = 0.0;
    std::vector<double> out;
    bool done = false;
    int steps = 0;
    for (int step = 0; step < steps_max; ++step) {
        basis.push_back(current);
        std::vector<Complex> next = multiply(current);
        Complex alpha_value{0.0, 0.0};
        for (int i = 0; i < n; ++i) alpha_value += std::conj(current[i]) * next[i];
        for (int i = 0; i < n; ++i) {
            next[i] -= alpha_value * current[i];
            next[i] -= beta_previous * previous[i];
        }
        // full reorthogonalisation, twice: the price of a trustworthy degenerate
        // spectrum, which is what a symmetry sector always has
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& keep : basis) {
                Complex overlap{0.0, 0.0};
                for (int i = 0; i < n; ++i) overlap += std::conj(keep[i]) * next[i];
                for (int i = 0; i < n; ++i) next[i] -= overlap * keep[i];
            }
        alpha[step] = alpha_value.real();
        if (step > 0) beta[step - 1] = beta_previous;
        double beta_value = 0.0;
        for (const auto& value : next) beta_value += std::norm(value);
        beta_value = std::sqrt(beta_value);
        ++steps;
        beta_end = beta_value;
        if (beta_value < tolerance) {
            // the Krylov space is invariant: the Ritz values are exact
            done = true;
        } else if (steps >= check_every && steps % check_every == 0) {
            out = ritz_values(alpha, beta, steps, count);
            done = converged(alpha, beta, steps, beta_end, out, tolerance);
        }
        if (done || step + 1 == steps_max) break;
        previous = std::move(current);
        beta_previous = beta_value;
        current = std::move(next);
        for (auto& value : current) value /= beta_value;
    }
    if (out.empty()) out = ritz_values(alpha, beta, steps, count);

    // a Ritz value is only an eigenvalue once its residual is small: the tail of
    // an unconverged recurrence produces numbers that lie between true levels and
    // are eigenvalues of nothing, and the caller is told rather than handed one
    if (!converged(alpha, beta, steps, beta_end, out, tolerance))
        throw std::runtime_error("Lanczos did not converge: raise the number of iterations");
    return out;
}

std::vector<double> lowest_eigenvalues(const SparseMatrix& matrix, int count,
                                       int iterations, double tolerance)
{
    if (matrix.rows != matrix.cols) throw std::invalid_argument("the matrix is not square");
    const int n = matrix.rows;
    return lowest_eigenvalues([&](const std::vector<Complex>& vector) {
        return multiply(matrix, vector);
    }, n, count, iterations, tolerance);
}

}  // namespace ed
