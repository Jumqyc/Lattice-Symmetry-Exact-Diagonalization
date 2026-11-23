#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hamiltonian.h"
#include "hilbert.h"
#include "lanczos.h"
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
        .def(py::init<int, int, std::vector<int>, int, bool>(),
             py::arg("sites"), py::arg("local_dim"), py::arg("charge"),
             py::arg("sztot") = -1, py::arg("with_table") = true)
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

    m.def("momentum_basis", &ed::momentum_basis,
          py::call_guard<py::gil_scoped_release>(), py::arg("basis"), py::arg("translations"),
          py::arg("characters"), py::arg("threads") = 0, "Momentum-adapted orthonormal basis of one sector");

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

    py::class_<ed::PhasedPermutation>(m, "PhasedPermutation")
        .def(py::init<std::vector<int>, std::vector<ed::Complex>>(),
             py::arg("act"), py::arg("phase"))
        .def_readonly("act", &ed::PhasedPermutation::act)
        .def_readonly("phase", &ed::PhasedPermutation::phase);

    m.def("sector_action", &ed::sector_action,
          py::call_guard<py::gil_scoped_release>(), py::arg("columns"), py::arg("basis"),
          py::arg("action"), "Action of one operation on a momentum sector");

    py::class_<ed::SectorVectors>(m, "SectorVectors")
        .def_readonly("start", &ed::SectorVectors::start)
        .def_readonly("columns", &ed::SectorVectors::columns)
        .def_readonly("data", &ed::SectorVectors::data)
        .def("vectors", &ed::SectorVectors::vectors);
    m.def("isotypic_basis", &ed::isotypic_basis,
          py::call_guard<py::gil_scoped_release>(), py::arg("actions"), py::arg("characters"),
          py::arg("dim"), py::arg("column_count"), py::arg("threads") = 0, "Adapted basis of one irrep");
    m.def("sector_hamiltonian", &ed::sector_hamiltonian,
          py::call_guard<py::gil_scoped_release>(), py::arg("basis"),
          py::arg("couplings"), py::arg("columns"), py::arg("threads") = 0,
          "The Hamiltonian in a momentum-adapted basis, term by term");
    py::class_<ed::SectorMap>(m, "SectorMap")
        .def_readonly("orbits", &ed::SectorMap::orbits,
                      "number of translation orbits in the sector")
        .def("__len__", [](const ed::SectorMap& map) { return map.orbits; });

    m.def("sector_map", &ed::sector_map, py::arg("basis"), py::arg("translations"),
          "Translated orbit labels of one momentum sector");
    m.def("sector_subset", &ed::sector_subset, py::arg("sector"), py::arg("characters"),
          "The same labels restricted to one momentum");
    m.def("action_on_orbits", &ed::action_on_orbits, py::arg("basis"),
          py::arg("sector"), py::arg("action"),
          "One little-group element acting on the orbit labels");
    m.def("sector_apply_block", &ed::sector_apply_block,
          py::call_guard<py::gil_scoped_release>(), py::arg("basis"),
          py::arg("couplings"), py::arg("sector"), py::arg("characters"),
          py::arg("vectors"), py::arg("vector"), py::arg("threads") = 0,
          "The Hamiltonian of one sector, applied matrix-free");
    m.def("sector_apply", &ed::sector_apply,
          py::call_guard<py::gil_scoped_release>(), py::arg("basis"),
          py::arg("couplings"), py::arg("sector"), py::arg("characters"),
          py::arg("vector"), py::arg("threads") = 0,
          "The sector Hamiltonian applied to a sector vector, matrix-free");
    m.def("lowest_eigenvalues_free",
          py::overload_cast<const ed::MatrixVector&, int, int, int, double, int>(
              &ed::lowest_eigenvalues),
          py::call_guard<py::gil_scoped_release>(), py::arg("multiply"),
          py::arg("dimension"), py::arg("count"), py::arg("iterations") = 200,
          py::arg("tolerance") = 1e-10, py::arg("threads") = 0,
          "Lowest eigenvalues of an operator given as a product");

    m.def("block_matrix", &ed::block_matrix,
          py::call_guard<py::gil_scoped_release>(), py::arg("matrix"), py::arg("vectors"),
          py::arg("threads") = 0,
          "The block of a sector in a set of adapted vectors, sparse");
    m.def("lowest_eigenvalues",
          py::overload_cast<const ed::SparseMatrix&, int, int, double, int>(&ed::lowest_eigenvalues),
          py::arg("matrix"), py::arg("count"),
          py::arg("iterations") = 200, py::arg("tolerance") = 1e-10,
          py::arg("threads") = 0);
}
