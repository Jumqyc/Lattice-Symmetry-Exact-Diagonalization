#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "group.h"
#include "hilbert.h"

namespace py = pybind11;

/**
 * @detail The kernel is deliberately free of any geometry: the multiplication
 *         table is built on the Python side, where the sample's own operation
 *         matrices and their comparison tolerance live, and the C++ side owns
 *         the algebra.  That keeps one source of truth for "when are two
 *         operations equal".
 * @short Python bindings of the group kernel.
 */
PYBIND11_MODULE(_edkern, m)
{
    m.doc() = "C++ kernel of the exact-diagonalisation symmetry machinery";

    py::class_<ed::CharacterTable>(m, "CharacterTable")
        .def_readonly("order", &ed::CharacterTable::order)
        .def_readonly("classes", &ed::CharacterTable::classes)
        .def_readonly("class_of", &ed::CharacterTable::class_of)
        .def_readonly("dims", &ed::CharacterTable::dims)
        .def_readonly("chars", &ed::CharacterTable::chars);

    m.def("character_table", &ed::character_table, py::arg("mult"),
          "Character table of a finite group given by its multiplication table");

    py::class_<ed::Basis>(m, "Basis")
        .def(py::init<int, int, std::vector<int>, int>(),
             py::arg("sites"), py::arg("local_dim"), py::arg("charge"),
             py::arg("sztot") = -1)
        .def("size", &ed::Basis::size)
        .def("sites", &ed::Basis::sites)
        .def("local_dim", &ed::Basis::local_dim)
        .def("state", &ed::Basis::state, py::arg("index"))
        .def("digit", &ed::Basis::digit, py::arg("word"), py::arg("site"))
        .def("index", &ed::Basis::index, py::arg("word"))
        .def("unpack", &ed::Basis::unpack, py::arg("word"));

    py::class_<ed::MonomialAction>(m, "MonomialAction")
        .def(py::init<std::vector<int>, std::vector<int>, std::vector<ed::Complex>>(),
             py::arg("site_perm"), py::arg("local_perm"), py::arg("local_phase"));

    m.def("apply", &ed::apply, py::arg("basis"), py::arg("action"), py::arg("word"),
          "U(g)|word> = (image, phase)");

    py::class_<ed::SparseColumns>(m, "SparseColumns")
        .def_readonly("reps", &ed::SparseColumns::reps)
        .def_readonly("indptr", &ed::SparseColumns::indptr)
        .def_readonly("indices", &ed::SparseColumns::indices)
        .def_readonly("data", &ed::SparseColumns::data)
        .def("columns", &ed::SparseColumns::columns);

    m.def("momentum_basis", &ed::momentum_basis, py::arg("basis"), py::arg("translations"),
          py::arg("characters"), "Momentum-adapted orthonormal basis of one sector");
}
