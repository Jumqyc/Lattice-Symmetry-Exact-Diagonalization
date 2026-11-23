#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "group.h"

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
}
