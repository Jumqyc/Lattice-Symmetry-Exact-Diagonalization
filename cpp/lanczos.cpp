#include "lanczos.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "hamiltonian.h"
#include "parallel.h"

namespace ed {

namespace {

/**
 * @param matrix the symmetric matrix, column-major, overwritten.
 * @param size its side.
 * @param values the eigenvalues, ascending.
 * @param vectors column-major; column ``i`` belongs to ``values[i]``.
 * @detail Cyclic Jacobi rotations.  The projected matrix is small -- one cycle
 *         plus the vectors kept across restarts, a few tens -- so rotating it
 *         directly is cheaper than the Stieltjes apparatus it replaces, and it
 *         hands back the eigenvectors that the restart needs in the same pass.
 * @short Eigenpairs of a small symmetric matrix.
 */
void jacobi(std::vector<double>& matrix, int size, std::vector<double>& values,
            std::vector<double>& vectors)
{
    vectors.assign(static_cast<std::size_t>(size) * size, 0.0);
    for (int i = 0; i < size; ++i) vectors[static_cast<std::size_t>(i) * size + i] = 1.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off_diagonal = 0.0;
        for (int p = 0; p < size; ++p)
            for (int q = p + 1; q < size; ++q) {
                const double value = matrix[p * size + q];
                off_diagonal += value * value;
            }
        if (off_diagonal <= 1e-30) break;
        for (int p = 0; p < size; ++p)
            for (int q = p + 1; q < size; ++q) {
                const double off = matrix[p * size + q];
                if (std::abs(off) <= 1e-18) continue;
                // the rotation that annihilates A[p][q], in the form that does not
                // overflow when the diagonal gap is small
                const double gap = 0.5 * (matrix[q * size + q] - matrix[p * size + p]);
                const double tangent = (gap >= 0.0 ? 1.0 : -1.0) * off /
                                       (std::abs(gap) + std::sqrt(gap * gap + off * off));
                const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
                const double sine = cosine * tangent;
                // A <- J^T A J, first the rows and then the columns
                for (int r = 0; r < size; ++r) {
                    const double left = matrix[r * size + p];
                    const double right = matrix[r * size + q];
                    matrix[r * size + p] = cosine * left - sine * right;
                    matrix[r * size + q] = sine * left + cosine * right;
                }
                for (int r = 0; r < size; ++r) {
                    const double left = matrix[p * size + r];
                    const double right = matrix[q * size + r];
                    matrix[p * size + r] = cosine * left - sine * right;
                    matrix[q * size + r] = sine * left + cosine * right;
                }
                for (int r = 0; r < size; ++r) {
                    const double left = vectors[r * size + p];
                    const double right = vectors[r * size + q];
                    vectors[r * size + p] = cosine * left - sine * right;
                    vectors[r * size + q] = sine * left + cosine * right;
                }
            }
    }
    std::vector<int> order(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return matrix[a * size + a] < matrix[b * size + b]; });
    std::vector<double> sorted_values(static_cast<std::size_t>(size));
    std::vector<double> sorted_vectors(static_cast<std::size_t>(size) * size);
    for (int i = 0; i < size; ++i) {
        sorted_values[i] = matrix[order[i] * size + order[i]];
        for (int r = 0; r < size; ++r) sorted_vectors[r * size + i] = vectors[r * size + order[i]];
    }
    values = std::move(sorted_values);
    vectors = std::move(sorted_vectors);
}

/**
 * @detail The recurrence itself, on a pool it does not own: the entry points
 *         below give it the same pool their product uses, so a step hands work to
 *         one set of threads instead of two.
 * @short Thick-restart Lanczos, with the caller's pool.
 */
std::vector<double> solve(const MatrixVector& multiply, int dimension, int count,
                          int iterations, double tolerance, Workers& pool)
{
    const int n = dimension;
    if (n <= 0) throw std::invalid_argument("the operator has no dimension");
    if (count <= 0 || count > n) throw std::invalid_argument("impossible number of eigenvalues");
    // A cycle is short on purpose.  Full reorthogonalisation reads every stored
    // vector at every step, so the cost of a step grows with the length of the
    // recurrence and the basis stops fitting in cache long before it stops
    // fitting in memory.  A cycle keeps a few tens of vectors; at its end the
    // wanted Ritz vectors are kept and the rest are dropped, which is the thick
    // restart: the kept vectors are exact Ritz vectors, so their residual is the
    // same direction (the next Lanczos vector) scaled by the last component of
    // their projected eigenvector, and the recurrence continues with the
    // projected matrix in arrowhead form.
    // The restart keeps more vectors than are wanted: keeping exactly the wanted
    // ones stalls on clustered spectra, because the recurrence is then restarted
    // into a subspace that no longer contains the directions the missing levels
    // need to grow in.
    const int kept_wanted = count + std::max(4, count);
    const int cycle = std::min(iterations, std::max(32, 2 * kept_wanted + 8));
    const int check_every = std::max(4, std::min(cycle, 10));

    // A phase is only worth spreading when it has enough work to pay for the
    // handshake: a dot product over ten thousand columns takes a few
    // microseconds, which is less than the barrier that would spread it.
    const auto worth_parallel = [&](std::size_t work) {
        return pool.size() > 1 && work >= 1u << 16;
    };
    const auto dot = [&](const std::vector<Complex>& left, const std::vector<Complex>& right) {
        if (!worth_parallel(static_cast<std::size_t>(n))) {
            Complex sum{0.0, 0.0};
            for (int i = 0; i < n; ++i) sum += std::conj(left[i]) * right[i];
            return sum;
        }
        std::vector<Complex> partial(static_cast<std::size_t>(pool.size()), Complex{0.0, 0.0});
        pool.run(n, [&](int index, int begin, int end) {
            Complex sum{0.0, 0.0};
            for (int i = begin; i < end; ++i) sum += std::conj(left[i]) * right[i];
            partial[index] = sum;
        });
        Complex total{0.0, 0.0};
        for (const Complex value : partial) total += value;
        return total;
    };
    const auto norm = [&](const std::vector<Complex>& value) {
        if (!worth_parallel(static_cast<std::size_t>(n))) {
            double sum = 0.0;
            for (const auto& entry : value) sum += std::norm(entry);
            return std::sqrt(sum);
        }
        std::vector<double> partial(static_cast<std::size_t>(pool.size()), 0.0);
        pool.run(n, [&](int index, int begin, int end) {
            double sum = 0.0;
            for (int i = begin; i < end; ++i) sum += std::norm(value[i]);
            partial[index] = sum;
        });
        double total = 0.0;
        for (const double entry : partial) total += entry;
        return std::sqrt(total);
    };
    const auto axpy = [&](std::vector<Complex>& target, Complex alpha,
                          const std::vector<Complex>& source) {
        const auto body = [&](int, int begin, int end) {
            for (int i = begin; i < end; ++i) target[i] -= alpha * source[i];
        };
        if (!worth_parallel(static_cast<std::size_t>(n))) body(0, 0, n);
        else pool.run(n, body);
    };
    const auto orthogonalise = [&](std::vector<Complex>& vector,
                                   const std::vector<std::vector<Complex>>& against) {
        if (against.empty()) return;
        std::vector<Complex> overlaps(against.size(), Complex{0.0, 0.0});
        const auto project = [&](int, int begin, int end) {
            for (int j = begin; j < end; ++j) {
                Complex sum{0.0, 0.0};
                for (int i = 0; i < n; ++i) sum += std::conj(against[j][i]) * vector[i];
                overlaps[j] = sum;
            }
        };
        if (!worth_parallel(against.size() * static_cast<std::size_t>(n)))
            project(0, 0, static_cast<int>(against.size()));
        else pool.run(static_cast<int>(against.size()), project);
        const auto subtract = [&](int, int begin, int end) {
            for (int i = begin; i < end; ++i) {
                Complex sum{0.0, 0.0};
                for (std::size_t j = 0; j < against.size(); ++j) sum += overlaps[j] * against[j][i];
                vector[i] -= sum;
            }
        };
        if (!worth_parallel(against.size() * static_cast<std::size_t>(n))) subtract(0, 0, n);
        else pool.run(n, subtract);
    };

    // deterministic pseudo-random start: a constant vector can be orthogonal to a
    // whole degenerate multiplet, which would hide it from the recurrence
    std::mt19937 rng(20261123u);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<Complex> current(static_cast<std::size_t>(n));
    for (auto& value : current) value = Complex{normal(rng), normal(rng)};
    const double start_norm = norm(current);
    for (auto& value : current) value /= start_norm;

    // the kept Ritz vectors with their values and their couplings to the first
    // vector of the new cycle, then the cycle's own vectors
    std::vector<std::vector<Complex>> kept;
    std::vector<double> kept_value;
    std::vector<double> kept_link;
    std::vector<std::vector<Complex>> fresh;
    std::vector<double> diagonal;
    std::vector<double> subdiagonal;

    std::vector<double> values;
    bool done = false;
    int total = 0;
    while (!done && total < iterations) {
        fresh.push_back(current);
        std::vector<Complex> next = multiply(current);
        const Complex rayleigh = dot(current, next);
        axpy(next, rayleigh, current);
        if (subdiagonal.empty() == false && fresh.size() >= 2)
            axpy(next, Complex{subdiagonal.back(), 0.0}, fresh[fresh.size() - 2]);
        // full reorthogonalisation, twice: the price of a trustworthy degenerate
        // spectrum, which is what a symmetry sector always has
        for (int pass = 0; pass < 2; ++pass) {
            orthogonalise(next, kept);
            orthogonalise(next, fresh);
        }
        diagonal.push_back(rayleigh.real());
        const double length = norm(next);
        subdiagonal.push_back(length);
        ++total;

        const bool checkpoint = static_cast<int>(fresh.size()) % check_every == 0 ||
                                static_cast<int>(fresh.size()) == cycle ||
                                total >= iterations || length <= tolerance;
        if (!checkpoint) {
            current = std::move(next);
            for (auto& value : current) value /= length;
            continue;
        }

        // the projected matrix: the kept block is diagonal, the first vector of
        // the cycle is coupled to all of them, and the rest is tridiagonal
        const int keep_count = static_cast<int>(kept.size());
        const int fresh_count = static_cast<int>(fresh.size());
        const int size = keep_count + fresh_count;
        std::vector<double> projected(static_cast<std::size_t>(size) * size, 0.0);
        for (int i = 0; i < keep_count; ++i) {
            projected[i * size + i] = kept_value[i];
            projected[i * size + keep_count] = kept_link[i];
            projected[keep_count * size + i] = kept_link[i];
        }
        for (int j = 0; j < fresh_count; ++j)
            projected[(keep_count + j) * size + (keep_count + j)] = diagonal[j];
        for (int j = 0; j + 1 < fresh_count; ++j) {
            const double coupling = subdiagonal[j];
            projected[(keep_count + j) * size + (keep_count + j + 1)] = coupling;
            projected[(keep_count + j + 1) * size + (keep_count + j)] = coupling;
        }
        std::vector<double> ritz;
        std::vector<double> ritz_vectors;
        jacobi(projected, size, ritz, ritz_vectors);

        // the residual of a Ritz pair is the first neglected coupling times the
        // last component of its projected eigenvector
        const int wanted = std::min(count, size);
        bool settled = size >= count;
        for (int i = 0; i < wanted && settled; ++i) {
            const double residual = std::abs(length * ritz_vectors[(size - 1) * size + i]);
            settled = residual <= tolerance * std::max(1.0, std::abs(ritz[i]));
        }
        if (settled) {
            values.assign(ritz.begin(), ritz.begin() + count);
            done = true;
            break;
        }
        current = std::move(next);
        if (length > tolerance) {
            for (auto& value : current) value /= length;
            if (static_cast<int>(fresh.size()) < cycle && total < iterations) continue;
        }

        // thick restart: the wanted Ritz vectors of this cycle become the kept
        // ones, and the vector after the last of them continues the recurrence
        const int keep = std::min(kept_wanted, size);
        std::vector<std::vector<Complex>> next_kept(static_cast<std::size_t>(keep));
        for (int i = 0; i < keep; ++i) {
            std::vector<Complex> combined(static_cast<std::size_t>(n), Complex{0.0, 0.0});
            for (int j = 0; j < keep_count; ++j) {
                const Complex weight = ritz_vectors[j * size + i];
                for (int a = 0; a < n; ++a) combined[a] += weight * kept[j][a];
            }
            for (int j = 0; j < fresh_count; ++j) {
                const Complex weight = ritz_vectors[(keep_count + j) * size + i];
                for (int a = 0; a < n; ++a) combined[a] += weight * fresh[j][a];
            }
            next_kept[i] = std::move(combined);
        }
        kept = std::move(next_kept);
        kept_value.assign(ritz.begin(), ritz.begin() + keep);
        kept_link.resize(static_cast<std::size_t>(keep));
        for (int i = 0; i < keep; ++i)
            kept_link[i] = length * ritz_vectors[(size - 1) * size + i];
        fresh.clear();
        diagonal.clear();
        subdiagonal.clear();
        if (length <= tolerance) {
            // A single start vector reaches one vector per *distinct* eigenvalue,
            // so a degenerate level closes the Krylov space with fewer vectors than
            // were asked for.  The remaining copies live in the orthogonal
            // complement: the recurrence continues from a fresh direction there,
            // which is what resolves them, one per cycle.
            for (auto& value : current) value = Complex{normal(rng), normal(rng)};
        }
        for (int pass = 0; pass < 2; ++pass) orthogonalise(current, kept);
        const double rescale = norm(current);
        if (rescale <= 1e-12)
            throw std::runtime_error("Lanczos lost its direction during a restart");
        for (auto& value : current) value /= rescale;
    }

    if (!done)
        throw std::runtime_error("Lanczos did not converge: raise the number of iterations");
    return values;
}

}  // namespace

std::vector<double> lowest_eigenvalues(const MatrixVector& multiply, int dimension,
                                       int count, int iterations, double tolerance,
                                       int threads)
{
    Workers pool(worker_count(dimension, threads));
    return solve(multiply, dimension, count, iterations, tolerance, pool);
}

std::vector<double> lowest_eigenvalues(const SparseMatrix& matrix, int count,
                                       int iterations, double tolerance, int threads)
{
    if (matrix.rows != matrix.cols) throw std::invalid_argument("the matrix is not square");
    const int n = matrix.rows;
    // one pool for the product and for the phases of the recurrence
    Workers pool(worker_count(n, threads));
    const MatrixVector product = [&](const std::vector<Complex>& vector) {
        return multiply(matrix, vector, &pool);
    };
    return solve(product, n, count, iterations, tolerance, pool);
}

}  // namespace ed
