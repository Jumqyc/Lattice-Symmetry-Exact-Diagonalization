'''
Representation theory: the character tables.

This module answers one question only -- *which irreducible representations
does this group have, and what are their characters*.  It says nothing about
the lattice and nothing about the Hilbert space.

Two kinds of group live here:

* :class:`TrivialGroup`, the group with one element;
* the helpers the modules above it share: removing duplicated operations, grouping
  them into cosets, and giving an operation a key that survives being written down
  and read back.

The 2D groups themselves -- cyclic and dihedral, named and unnamed -- live in
``geometry/2dsym.py``, which is where the angles are.

Groups with a standard name (C4, D6, ...) live in ``geometry/2dsym.py``
together with their hand-written tables.
'''

from __future__ import annotations

import numpy as np

from base import PointGroup


# operations of a point group are compared up to this tolerance
_OP_TOL = 1e-8
_OP_TOL_DIGITS = 12


def _matrix_to_flat(op: np.ndarray) -> tuple[int | float, ...]:
    '''
    Convert a matrix to a flattened tuple suitable for hashing.

    Entries within ``_OP_TOL`` of an integer are stored as ``int``;
    the others are rounded to ``_OP_TOL_DIGITS`` decimal places.

    Args:
        op: a square matrix, typically a point-group operation.
    Returns:
        A hashable, rounding-insensitive representation of the matrix.
    '''
    flat: list[int | float] = []
    for x in np.asarray(op, dtype=float).ravel():
        r = round(float(x))
        if abs(x - r) < _OP_TOL:
            flat.append(int(r))
        else:
            flat.append(round(float(x), _OP_TOL_DIGITS))
    return tuple(flat)


def _op_key(op: np.ndarray) -> tuple[float, ...]:
    '''Rounding-insensitive hashable key of an operation matrix.'''
    return tuple(np.round(np.asarray(op, dtype=float).ravel(), _OP_TOL_DIGITS))


def _unique_ops(ops: list[np.ndarray]) -> list[np.ndarray]:
    '''Drop duplicated operations, keeping the order of first appearance.'''
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for op in ops:
        unique.setdefault(_op_key(op), np.asarray(op, dtype=float))
    return list(unique.values())


def _right_coset_representatives(ops: list[np.ndarray],
                                 subgroup: list[np.ndarray]
                                 ) -> list[np.ndarray]:
    '''
    One representative per right coset ``G_k \\ G_pt``.

    Args:
        ops: the operations of the full group, without duplicates.
        subgroup: the operations of the subgroup, without duplicates.
    Returns:
        A list with ``|G_pt| / |G_k|`` operations, one from each coset
        ``G_k g``.
    Raises:
        ValueError: if *subgroup* is not a subgroup of *ops*.
    '''
    index = {_op_key(op) for op in ops}
    reps: list[np.ndarray] = []
    cosets: list[frozenset[tuple[float, ...]]] = []
    for g in ops:
        coset = frozenset(_op_key(h @ g) for h in subgroup)
        if not coset <= index:
            raise ValueError("The subgroup is not a subgroup of the point group.")
        if coset not in cosets:
            cosets.append(coset)
            reps.append(g)
    return reps


class TrivialGroup(PointGroup):
    '''
    A trivial point group containing only the identity operation.

    The only irrep of this group is 'A', which has a character of 1 for the identity operation.
    '''
    def __init__(self):
        # the identity is the 2x2 identity: every operation of this package acts on
        # the plane, and the little-group machinery composes and multiplies them,
        # which a bare scalar cannot take part in
        self._shape: tuple[int, ...] = (2, 2)
        self._operations: tuple[tuple[int | float, ...], ...] = ((1, 0, 0, 1),)

    def __hash__(self):
        return hash((self._shape, self._operations))

    def __eq__(self, other):
        if not isinstance(other, TrivialGroup):
            return NotImplemented
        return True

    def _irrep_labels(self) -> list[str]:
        return ['A']

    def _character(self, irrep: str, op: np.ndarray) -> complex:
        if irrep == 'A':
            return 1.0
        else:
            raise ValueError(f"Unknown irreducible representation: {irrep}")

    def __iter__(self):
        return iter([np.array(self._operations[0], dtype=np.float64)
                     .reshape(self._shape)])
