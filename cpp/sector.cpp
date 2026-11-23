#include "sector.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#include <Eigen/Dense>

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
    for (int index = 0; index < static_cast<int>(orbits.size()); ++index) {
        const std::vector<int>& orbit = orbits[index];
        const int size = static_cast<int>(orbit.size());
        std::vector<int> local(column_count, -1);
        for (int i = 0; i < size; ++i) local[orbit[i]] = i;

        // the projector on this orbit: a sum of |G| phased permutation matrices,
        // reduced to the size of the orbit instead of the size of the sector
        Eigen::MatrixXcd projector = Eigen::MatrixXcd::Zero(size, size);
        for (int g = 0; g < groups; ++g)
            for (int i = 0; i < size; ++i) {
                const int to = local[actions[g].act[orbit[i]]];
                projector(to, i) += std::conj(characters[g]) * actions[g].phase[orbit[i]]
                                    * (static_cast<double>(dim) / static_cast<double>(groups));
            }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(projector);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("the sector projector could not be diagonalised");
        for (int rho = 0; rho < size; ++rho) {
            if (solver.eigenvalues()[rho] < 0.5) continue;
            const Eigen::VectorXcd vector = solver.eigenvectors().col(rho);
            for (int i = 0; i < size; ++i) {
                if (std::abs(vector[i]) < 1e-12) continue;
                out.columns.push_back(orbit[i]);
                out.data.push_back(vector[i]);
            }
            out.start.push_back(static_cast<int>(out.columns.size()));
            out.orbit_of_vector.push_back(index);
        }
    }
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

std::vector<Complex> block_matrix(const SparseMatrix& matrix, const SectorVectors& vectors)
{
    const int m = vectors.vectors();
    if (m == 0) return {};

    // V^T H V, one column at a time: H applied to the vector, then read back on
    // the support of every vector.  A dense scratch vector of the size of the
    // sector is the cheapest way to say this unambiguously -- the previous
    // hand-written merge of two sorted lists was wrong for sectors whose rows
    // are hit several times.
    std::vector<Complex> out(static_cast<std::size_t>(m) * m, Complex{0.0, 0.0});
    std::vector<Complex> applied(static_cast<std::size_t>(matrix.cols));
    for (int c = 0; c < m; ++c) {
        std::fill(applied.begin(), applied.end(), Complex{0.0, 0.0});
        for (int a = vectors.start[c]; a < vectors.start[c + 1]; ++a) {
            const int source = vectors.columns[a];
            const Complex weight = vectors.data[a];
            for (int b = matrix.indptr[source]; b < matrix.indptr[source + 1]; ++b)
                applied[matrix.indices[b]] += matrix.data[b] * weight;
        }
        for (int r = 0; r < m; ++r) {
            Complex sum{0.0, 0.0};
            for (int a = vectors.start[r]; a < vectors.start[r + 1]; ++a)
                sum += std::conj(vectors.data[a]) * applied[vectors.columns[a]];
            out[static_cast<std::size_t>(r) + static_cast<std::size_t>(m) * c] = sum;
        }
    }
    return out;
}

std::vector<double> eigenvalues(const std::vector<Complex>& block, int size)
{
    if (size <= 0) return {};          // an irrep can be absent from a sector
    Eigen::MatrixXcd matrix(size, size);
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            matrix(i, j) = block[static_cast<std::size_t>(i) + static_cast<std::size_t>(size) * j];
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(matrix);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("the block could not be diagonalised");
    std::vector<double> out(size);
    for (int i = 0; i < size; ++i) out[i] = solver.eigenvalues()[i];
    return out;
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

    std::vector<std::vector<Complex>> basis;
    basis.reserve(steps_max);
    Eigen::MatrixXd tridiagonal = Eigen::MatrixXd::Zero(steps_max, steps_max);
    std::vector<Complex> previous(static_cast<std::size_t>(n), Complex{0.0, 0.0});
    double beta_previous = 0.0;
    int steps = 0;
    for (int step = 0; step < steps_max; ++step) {
        basis.push_back(current);
        std::vector<Complex> next = multiply(matrix, current);
        Complex alpha{0.0, 0.0};
        for (int i = 0; i < n; ++i) alpha += std::conj(current[i]) * next[i];
        for (int i = 0; i < n; ++i) {
            next[i] -= alpha * current[i];
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
        tridiagonal(step, step) = alpha.real();
        if (step > 0) {
            tridiagonal(step - 1, step) = beta_previous;
            tridiagonal(step, step - 1) = beta_previous;
        }
        double beta = 0.0;
        for (const auto& value : next) beta += std::norm(value);
        beta = std::sqrt(beta);
        ++steps;
        if (beta < tolerance || step + 1 == steps_max) break;
        previous = std::move(current);
        beta_previous = beta;
        current = std::move(next);
        for (auto& value : current) value /= beta;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(tridiagonal.topLeftCorner(steps, steps));
    const Eigen::VectorXd values = solver.eigenvalues();
    std::vector<double> out(std::min<int>(count, static_cast<int>(values.size())));
    for (int i = 0; i < static_cast<int>(out.size()); ++i) out[i] = values[i];
    return out;
}

}  // namespace ed
