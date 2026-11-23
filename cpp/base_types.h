#pragma once

#include <complex>
#include <vector>

namespace ed {

using Complex = std::complex<double>;

/**
 * @detail CSR storage, the only matrix form the kernel ever needs: the
 *         Hamiltonian, the momentum basis (as columns) and the sector blocks
 *         are all sparse, and the dense form appears only for the small block
 *         that is handed to the eigensolver.
 * @short A sparse matrix in compressed-row form.
 */
struct SparseMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> indptr;      // size rows + 1
    std::vector<int> indices;     // column index of each entry
    std::vector<Complex> data;    // value of each entry

    int nonzeros() const { return static_cast<int>(data.size()); }
};

}  // namespace ed
