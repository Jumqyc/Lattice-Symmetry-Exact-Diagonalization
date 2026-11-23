#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hamiltonian.h"
#include "hilbert.h"
#include "sector.h"

namespace py = pybind11;

/**
 * @detail The kernel is deliberately free of geometry and of group theory: the
 *         sites, the operations, the little groups and their characters are all
 *         built on the Python side, and the C++ side owns only the objects that
 *         scale with the Hilbert space.  What crosses the boundary is a model as
 *         plain arrays, and a sector decomposition as sparse matrices.
 * @short Python bindings of the kernel.
 */
PYBIND11_MODULE(_edkern, m)
{
    m.doc() = "C++ kernel of exact diagonalisation: states, actions, sectors";

    py::class_<ed::Basis>(m, "Basis")
        .def(py::init<int, int, std::vector<int>, int>(),
             py::arg("sites"), py::arg("local_dim"), py::arg("charge"),
             py::arg("sztot") = -1)
        .def("size", &ed::Basis::size, "number of states")
        .def("state", &ed::Basis::state, py::arg("index"), "the packed state")
        .def("index", &ed::Basis::index, py::arg("word"), "index of a packed state")
        .def("unpack", &ed::Basis::unpack, py::arg("word"), "the state as digit tuples");

    py::class_<ed::MonomialAction>(m, "MonomialAction")
        .def(py::init<std::vector<int>, std::vector<int>, std::vector<ed::Complex>>(),
             py::arg("site_perm"), py::arg("local_perm"), py::arg("local_phase"));

    m.def("apply", &ed::apply, py::arg("basis"), py::arg("action"), py::arg("word"),
          "U(g)|word> = (image, phase)");

    py::class_<ed::SparseColumns>(m, "SparseColumns")
        .def_readonly("indptr", &ed::SparseColumns::indptr)
        .def_readonly("indices", &ed::SparseColumns::indices)
        .def_readonly("data", &ed::SparseColumns::data)
        .def("columns", &ed::SparseColumns::columns);

    m.def("momentum_basis", &ed::momentum_basis, py::arg("basis"), py::arg("translations"),
          py::arg("characters"), "Momentum-adapted orthonormal basis of one sector");

    py::class_<ed::SparseMatrix>(m, "SparseMatrix")
        .def_readonly("rows", &ed::SparseMatrix::rows)
        .def_readonly("cols", &ed::SparseMatrix::cols)
        .def_readonly("indptr", &ed::SparseMatrix::indptr)
        .def_readonly("indices", &ed::SparseMatrix::indices)
        .def_readonly("data", &ed::SparseMatrix::data)
        .def("nonzeros", &ed::SparseMatrix::nonzeros);

    py::class_<ed::LocalTerm>(m, "LocalTerm")
        .def(py::init<std::vector<int>, std::vector<int>, std::vector<ed::Complex>>(),
             py::arg("out"), py::arg("in"), py::arg("coeff"));

    py::class_<ed::CouplingTerm>(m, "CouplingTerm")
        .def(py::init<std::vector<int>, std::vector<ed::LocalTerm>>(),
             py::arg("sites"), py::arg("local"));

    m.def("assemble", &ed::assemble, py::arg("basis"), py::arg("couplings"),
          "The Hamiltonian in the computational basis, in CSR form");
    m.def("symmetrise", &ed::symmetrise, py::arg("matrix"), py::arg("basis"),
          py::arg("actions"), "Symmetric part of a Hamiltonian");

    py::class_<ed::PhasedPermutation>(m, "PhasedPermutation")
        .def(py::init<std::vector<int>, std::vector<ed::Complex>>(),
             py::arg("act"), py::arg("phase"))
        .def_readonly("act", &ed::PhasedPermutation::act)
        .def_readonly("phase", &ed::PhasedPermutation::phase);

    m.def("sector_action", &ed::sector_action, py::arg("columns"), py::arg("basis"),
          py::arg("action"), "Action of one operation on a momentum sector");

    py::class_<ed::SectorVectors>(m, "SectorVectors")
        .def_readonly("start", &ed::SectorVectors::start)
        .def_readonly("columns", &ed::SectorVectors::columns)
        .def_readonly("data", &ed::SectorVectors::data)
        .def("vectors", &ed::SectorVectors::vectors);

    m.def("identity_vectors", &ed::identity_vectors, py::arg("column_count"),
          "The sector basis itself, for a trivial little group");
    m.def("isotypic_basis", &ed::isotypic_basis, py::arg("actions"), py::arg("characters"),
          py::arg("dim"), py::arg("column_count"), "Adapted basis of one irrep");
    m.def("sector_hamiltonian", &ed::sector_hamiltonian, py::arg("hamiltonian"),
          py::arg("columns"), "The Hamiltonian in a momentum-adapted basis");
    m.def("block_matrix", &ed::block_matrix, py::arg("matrix"), py::arg("vectors"),
          "The block of a sector in a set of adapted vectors, sparse");
    m.def("lowest_eigenvalues", &ed::lowest_eigenvalues, py::arg("matrix"), py::arg("count"),
          py::arg("iterations") = 200, py::arg("tolerance") = 1e-10);
}
