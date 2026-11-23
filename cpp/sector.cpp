#include "sector.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace ed {

namespace {

/**
 * @detail Groups the columns of a sector into orbits of the little group.  The
 *         action is a permutation, so a plain walk marks each orbit once; the
 *         result is a list of orbits, each a list of column indices.
 * @short Orbits of the little group on the sector columns.
 */
std::vector<std::vector<int>> column_orbits(const std::vector<PhasedPermutation>& actions,
                                            int count)
{
    std::vector<char> seen(count, 0);
    std::vector<std::vector<int>> orbits;
    std::vector<int> stack;
    for (int start = 0; start < count; ++start) {
        if (seen[start]) continue;
        std::vector<int> orbit;
        stack.assign(1, start);
        seen[start] = 1;
        while (!stack.empty()) {
            const int x = stack.back();
            stack.pop_back();
            orbit.push_back(x);
            for (const PhasedPermutation& action : actions) {
                const int next = action.act[x];
                if (!seen[next]) {
                    seen[next] = 1;
                    stack.push_back(next);
                }
            }
        }
        std::sort(orbit.begin(), orbit.end());
        orbits.push_back(orbit);
    }
    return orbits;
}

/**
 * @param alpha diagonal of the tridiagonal Lanczos matrix.
 * @param beta subdiagonal; ``beta[i]`` joins step ``i`` and step ``i+1``.
 * @param x the shift.
 * @detail The number of negative pivots of the LDL^T factor of ``T - x`` equals
 *         the number of eigenvalues below ``x`` (Sylvester's law of inertia).
 *         Only the two diagonals are touched, so no matrix is formed.
 * @short How many eigenvalues of the tridiagonal lie below ``x``.
 */
int count_below(const std::vector<double>& alpha, const std::vector<double>& beta, double x)
{
    int below = 0;
    double pivot = alpha[0] - x;
    if (pivot < 0.0) ++below;
    for (std::size_t i = 1; i < alpha.size(); ++i) {
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
                      double theta)
{
    const int steps = static_cast<int>(alpha.size());
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

}  // namespace

PhasedPermutation sector_action(const SparseColumns& columns, const Basis& basis,
                                const MonomialAction& action)
{
    const int n = basis.size();
    // state -> column of the sector; every state belongs to exactly one column
    std::vector<int> column_of(n, -1);
    for (int c = 0; c < columns.columns(); ++c)
        for (int a = columns.indptr[c]; a < columns.indptr[c + 1]; ++a)
            column_of[columns.indices[a]] = c;

    PhasedPermutation out;
    out.act.resize(columns.columns());
    out.phase.resize(columns.columns());
    for (int c = 0; c < columns.columns(); ++c) {
        const int probe = columns.indices[columns.indptr[c]];
        const Complex probe_amplitude = columns.data[columns.indptr[c]];
        const auto moved = apply(basis, action, basis.state(probe));
        const int image = basis.index(moved.first);
        if (image < 0 || column_of[image] < 0)
            throw std::runtime_error("the operation leaves the momentum sector");
        const int target = column_of[image];
        // coefficient of the image in its own column
        Complex image_amplitude{0.0, 0.0};
        for (int a = columns.indptr[target]; a < columns.indptr[target + 1]; ++a)
            if (columns.indices[a] == image) image_amplitude = columns.data[a];
        if (std::abs(image_amplitude) < 1e-14)
            throw std::runtime_error("the image of a probe has no amplitude");
        out.act[c] = target;
        out.phase[c] = probe_amplitude * moved.second / image_amplitude;
    }
    return out;
}

SectorVectors isotypic_basis(const std::vector<PhasedPermutation>& actions,
                             const std::vector<Complex>& characters,
                             int dim, int column_count)
{
    const int groups = static_cast<int>(characters.size());
    if (static_cast<int>(actions.size()) != groups)
        throw std::invalid_argument("one action per character is required");
    const std::vector<std::vector<int>> orbits = column_orbits(actions, column_count);

    SectorVectors out;
    out.start.assign(1, 0);
    std::vector<Complex> candidate;             // one orbit, never the sector
    for (int index = 0; index < static_cast<int>(orbits.size()); ++index) {
        const std::vector<int>& orbit = orbits[index];
        const int size = static_cast<int>(orbit.size());
        std::vector<int> local(column_count, -1);
        for (int i = 0; i < size; ++i) local[orbit[i]] = i;

        // the rank the character predicts on this orbit, from the trace of the
        // projector: it is what the orthonormalisation below has to reproduce
        Complex trace{0.0, 0.0};
        for (int g = 0; g < groups; ++g) {
            Complex fixed{0.0, 0.0};
            for (int i = 0; i < size; ++i)
                if (local[actions[g].act[orbit[i]]] == i) fixed += actions[g].phase[orbit[i]];
            trace += std::conj(characters[g]) * fixed;
        }
        const int expected = static_cast<int>(std::lround(
            (static_cast<double>(dim) / static_cast<double>(groups)) * trace.real()));

        // the projected basis vectors of this orbit, orthonormalized one at a
        // time: the image of the projector is spanned by them, so this gives an
        // orthonormal basis of the isotypic component of the orbit
        std::vector<std::vector<std::pair<int, Complex>>> accepted;
        for (int x = 0; x < size && static_cast<int>(accepted.size()) < expected; ++x) {
            candidate.assign(size, Complex{0.0, 0.0});
            for (int g = 0; g < groups; ++g) {
                const int to = local[actions[g].act[orbit[x]]];
                candidate[to] += std::conj(characters[g]) * actions[g].phase[orbit[x]]
                                 * (static_cast<double>(dim) / static_cast<double>(groups));
            }
            // two passes of modified Gram-Schmidt over the accepted supports; a
            // single pass loses orthogonality as soon as the orbit is larger
            // than a handful of columns
            for (int pass = 0; pass < 2; ++pass)
                for (const auto& keep : accepted) {
                    Complex overlap{0.0, 0.0};
                    for (const auto& entry : keep)
                        overlap += std::conj(entry.second) * candidate[entry.first];
                    for (const auto& entry : keep)
                        candidate[entry.first] -= overlap * entry.second;
                }
            double norm = 0.0;
            for (const Complex value : candidate) norm += std::norm(value);
            norm = std::sqrt(norm);
            if (norm < 1e-8) continue;          // dependent on what is already in
            std::vector<std::pair<int, Complex>> vector;
            for (int i = 0; i < size; ++i) {
                if (std::abs(candidate[i]) < 1e-14) continue;
                vector.emplace_back(i, candidate[i] / norm);
            }
            accepted.push_back(std::move(vector));
        }
        if (static_cast<int>(accepted.size()) != expected)
            throw std::runtime_error(
                "the isotypic component of an orbit has the wrong dimension: the "
                "character table and the action on the sector disagree");

        for (const auto& vector : accepted) {
            for (const auto& entry : vector) out.columns.push_back(orbit[entry.first]);
            for (const auto& entry : vector) out.data.push_back(entry.second);
            out.start.push_back(static_cast<int>(out.columns.size()));
            out.orbit_of_vector.push_back(index);
        }
    }
    return out;
}

SectorVectors identity_vectors(int column_count)
{
    SectorVectors out;
    out.start.resize(static_cast<std::size_t>(column_count) + 1);
    out.columns.resize(column_count);
    out.data.assign(column_count, Complex{1.0, 0.0});
    out.orbit_of_vector.resize(column_count);
    for (int c = 0; c < column_count; ++c) {
        out.start[c] = c;
        out.columns[c] = c;
        out.orbit_of_vector[c] = c;
    }
    out.start[column_count] = column_count;
    return out;
}

SparseMatrix sector_hamiltonian(const SparseMatrix& hamiltonian, const SparseColumns& columns)
{
    const int n = hamiltonian.rows;
    const int count = columns.columns();
    if (hamiltonian.cols != n) throw std::invalid_argument("the Hamiltonian is not square");

    // both factors are walked by column, so both matrices are transposed once
    const SparseMatrix hamiltonian_t = transpose(hamiltonian);
    SparseMatrix columns_t;                 // rows are states, columns are sector columns
    columns_t.rows = n;
    columns_t.cols = count;
    columns_t.indptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int c = 0; c < count; ++c)
        for (int a = columns.indptr[c]; a < columns.indptr[c + 1]; ++a)
            ++columns_t.indptr[columns.indices[a] + 1];
    for (int i = 0; i < n; ++i) columns_t.indptr[i + 1] += columns_t.indptr[i];
    columns_t.indices.resize(columns.data.size());
    columns_t.data.resize(columns.data.size());
    std::vector<int> cursor(columns_t.indptr.begin(), columns_t.indptr.end() - 1);
    for (int c = 0; c < count; ++c)
        for (int a = columns.indptr[c]; a < columns.indptr[c + 1]; ++a) {
            const int at = cursor[columns.indices[a]]++;
            columns_t.indices[at] = c;
            columns_t.data[at] = columns.data[a];
        }

    // the accumulation below runs column by column, so the result is built in
    // compressed-column form and transposed at the end
    SparseMatrix by_column;
    by_column.rows = count;
    by_column.cols = count;
    by_column.indptr.assign(static_cast<std::size_t>(count) + 1, 0);
    std::vector<Complex> row(count);
    std::vector<std::pair<int, Complex>> product;   // H B, column c: (state row, value)
    for (int c = 0; c < count; ++c) {
        // Y = H B, column c
        product.clear();
        for (int a = columns.indptr[c]; a < columns.indptr[c + 1]; ++a) {
            const int state = columns.indices[a];
            const Complex weight = columns.data[a];
            for (int b = hamiltonian_t.indptr[state]; b < hamiltonian_t.indptr[state + 1]; ++b)
                product.emplace_back(hamiltonian_t.indices[b], hamiltonian_t.data[b] * weight);
        }
        // B^T Y, column c
        std::fill(row.begin(), row.end(), Complex{0.0, 0.0});
        for (const auto& entry : product) {
            for (int b = columns_t.indptr[entry.first]; b < columns_t.indptr[entry.first + 1]; ++b)
                row[columns_t.indices[b]] += std::conj(columns_t.data[b]) * entry.second;
        }
        for (int r = 0; r < count; ++r) {
            if (std::abs(row[r]) < 1e-14) continue;
            by_column.indices.push_back(r);
            by_column.data.push_back(row[r]);
        }
        by_column.indptr[c + 1] = static_cast<int>(by_column.indices.size());
    }
    return transpose(by_column);
}

SparseMatrix block_matrix(const SparseMatrix& matrix, const SectorVectors& vectors)
{
    const int m = vectors.vectors();
    SparseMatrix by_column;
    by_column.rows = m;
    by_column.cols = m;
    by_column.indptr.assign(static_cast<std::size_t>(m) + 1, 0);
    if (m == 0) return by_column;

    // V^T, walked by sector column: the block rows whose vector contains that
    // column.  A vector lives on one orbit, so this stays as sparse as V.
    std::vector<int> start(static_cast<std::size_t>(matrix.cols) + 1, 0);
    for (const int column : vectors.columns) ++start[column + 1];
    for (int a = 0; a < matrix.cols; ++a) start[a + 1] += start[a];
    std::vector<int> row_of(vectors.columns.size());
    std::vector<Complex> value_of(vectors.columns.size());
    std::vector<int> cursor(start.begin(), start.end() - 1);
    for (int c = 0; c < m; ++c)
        for (int a = vectors.start[c]; a < vectors.start[c + 1]; ++a) {
            const int at = cursor[vectors.columns[a]]++;
            row_of[at] = c;
            value_of[at] = std::conj(vectors.data[a]);
        }

    // V^T H V, one column at a time: H applied to the column of V stays sparse,
    // and every entry of it is scattered over the few block rows that contain
    // its sector row.  Nothing of the size of the sector is ever allocated.
    // The product walks the columns of H, so H is transposed once: walking its
    // rows instead would accumulate (H^T v).
    const SparseMatrix transposed = transpose(matrix);
    std::vector<std::pair<int, Complex>> applied;
    std::vector<std::pair<int, Complex>> terms;
    for (int c = 0; c < m; ++c) {
        applied.clear();
        for (int a = vectors.start[c]; a < vectors.start[c + 1]; ++a) {
            const int source = vectors.columns[a];
            const Complex weight = vectors.data[a];
            for (int b = transposed.indptr[source]; b < transposed.indptr[source + 1]; ++b)
                applied.emplace_back(transposed.indices[b], transposed.data[b] * weight);
        }
        std::sort(applied.begin(), applied.end(),
                  [](const auto& left, const auto& right) { return left.first < right.first; });
        terms.clear();
        for (std::size_t i = 0; i < applied.size();) {
            const int row = applied[i].first;
            Complex sum{0.0, 0.0};
            std::size_t j = i;
            while (j < applied.size() && applied[j].first == row) {
                sum += applied[j].second;
                ++j;
            }
            i = j;
            if (std::abs(sum) < 1e-14) continue;
            for (int t = start[row]; t < start[row + 1]; ++t)
                terms.emplace_back(row_of[t], value_of[t] * sum);
        }
        std::sort(terms.begin(), terms.end(),
                  [](const auto& left, const auto& right) { return left.first < right.first; });
        for (std::size_t i = 0; i < terms.size();) {
            const int row = terms[i].first;
            Complex sum{0.0, 0.0};
            std::size_t j = i;
            while (j < terms.size() && terms[j].first == row) {
                sum += terms[j].second;
                ++j;
            }
            i = j;
            if (std::abs(sum) < 1e-14) continue;
            by_column.indices.push_back(row);
            by_column.data.push_back(sum);
        }
        by_column.indptr[c + 1] = static_cast<int>(by_column.indices.size());
    }
    return transpose(by_column);
}

std::vector<double> lowest_eigenvalues(const SparseMatrix& matrix, int count,
                                       int iterations, double tolerance)
{
    const int n = matrix.rows;
    if (matrix.cols != n) throw std::invalid_argument("the matrix is not square");
    if (count <= 0 || count > n) throw std::invalid_argument("impossible number of eigenvalues");
    // with as many steps as the dimension the recurrence spans the whole space
    // and the Ritz values are exact; fewer steps give the usual approximation
    const int steps_max = std::min(iterations, n);

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
    int steps = 0;
    for (int step = 0; step < steps_max; ++step) {
        basis.push_back(current);
        std::vector<Complex> next = multiply(matrix, current);
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
        if (beta_value < tolerance || step + 1 == steps_max) break;
        previous = std::move(current);
        beta_previous = beta_value;
        current = std::move(next);
        for (auto& value : current) value /= beta_value;
    }
    alpha.resize(steps);
    if (steps > 1) beta.resize(steps - 1);

    // Spectral bounds from the Gershgorin discs of the tridiagonal.  Every row
    // has two off-diagonal entries except the first and the last, so the bound
    // is the row sum and not a single beta: a bracket that cuts off the two
    // lowest levels makes the bisection converge to the edge of the bracket and
    // report a value that is not an eigenvalue at all.
    double low = alpha[0] - std::abs(beta.empty() ? 0.0 : beta[0]);
    double high = alpha[steps - 1] + (steps > 1 ? std::abs(beta[steps - 2]) : 0.0);
    for (int i = 0; i < steps; ++i) {
        const double left = i > 0 ? std::abs(beta[i - 1]) : 0.0;
        const double right = i + 1 < steps ? std::abs(beta[i]) : 0.0;
        low = std::min(low, alpha[i] - left - right);
        high = std::max(high, alpha[i] + left + right);
    }
    const double resolution = (high - low) * 1e-15;

    // the lowest Ritz values, by bisection on the inertia count: the tridiagonal
    // is never stored, let alone diagonalised as a dense matrix
    std::vector<double> out(static_cast<std::size_t>(count), 0.0);
    for (int j = 0; j < count; ++j) {
        double left = low;
        double right = high;
        for (int iteration = 0; iteration < 200 && (right - left) > resolution; ++iteration) {
            const double middle = 0.5 * (left + right);
            if (count_below(alpha, beta, middle) > j) right = middle;
            else left = middle;
        }
        out[j] = 0.5 * (left + right);
    }

    // A Ritz value is only an eigenvalue once its residual is small; the tail of
    // an unconverged recurrence produces numbers that lie between true levels
    // and are eigenvalues of nothing.  The residual of a Ritz pair is the last
    // component of its tridiagonal eigenvector times the first neglected
    // off-diagonal, so it costs one pass over the recurrence.
    for (int j = 0; j < count; ++j) {
        const double residual = beta_end * last_component(alpha, beta, out[j]);
        if (residual > tolerance * std::max(1.0, std::abs(out[j])))
            throw std::runtime_error(
                "Lanczos did not converge: raise the number of iterations");
    }
    return out;
}

}  // namespace ed
