'''
Representation theory: the character tables.

This module answers one question only -- *which irreducible representations
does this group have, and what are their characters*.  It says nothing about
the lattice and nothing about the Hilbert space.

Two kinds of group live here:

* :class:`TrivialGroup`, the group with one element;
* :class:`ExplicitPointGroup`, a group given by its list of operation
  matrices, with the character table derived from the group's structure.
  This is what the little groups are built from, since a little group is a
  generic subgroup and has no dedicated class.

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

    def all_irrps(self) -> list[str]:
        return ['A']

    def character(self, irrep:str, op:np.ndarray) -> complex:
        if irrep == 'A':
            return 1.0
        else:
            raise ValueError(f"Unknown irreducible representation: {irrep}")

    def __iter__(self):
        return iter([np.array(self._operations[0], dtype=np.float64)
                     .reshape(self._shape)])


class ExplicitPointGroup(PointGroup):
    '''
    A point group given explicitly by its list of operation matrices.

    This is the class used for the little groups, which are generic subgroups
    of the point group and consequently have no dedicated class.  Only the
    operation list is needed: any finite subgroup of O(2) is cyclic C_n or
    dihedral D_n, so the irreps follow from the order *n* of the rotation
    subgroup and from whether the group contains reflections.

    Irreps
    ------
    * C_n: n one-dimensional irreps ``'j0' … 'j{n-1}'`` with
      ``χ_j(r^m) = exp(2πi j m / n)``.
    * D_n: ``'A1'``, ``'A2'`` (plus ``'B1'``, ``'B2'`` for even n) and
      ``(n-1)//2`` two-dimensional irreps ``'E1' …`` with
      ``χ_{E_j}(r^m) = 2 cos(2π j m / n)`` on rotations and 0 on reflections.
      A reflection is written ``σ = σ_0 r^m`` with σ_0 the first reflection of
      the list; its character is ``(-1)^m`` for ``B1`` and ``-(-1)^m`` for
      ``B2``.

    The operations must form a group.  This is checked on construction, because
    a set of matrices that is not closed silently produces a meaningless
    character table and therefore meaningless sectors.
    '''

    def __init__(self, operations: list[np.ndarray]):
        ops = _unique_ops(operations)
        if not ops:
            raise ValueError("A point group needs at least one operation.")
        shape = ops[0].shape
        if shape != (2, 2):
            raise ValueError(
                "ExplicitPointGroup knows the irreps of subgroups of O(2) only, "
                f"got operations of shape {shape}.")
        for op in ops:
            if not np.allclose(op @ op.T, np.eye(2), atol=_OP_TOL):
                raise ValueError("The operations must be orthogonal matrices.")

        # closure: every product of two operations must be one of them
        keys = {_op_key(op) for op in ops}
        for a in ops:
            for b in ops:
                if _op_key(a @ b) not in keys:
                    raise ValueError(
                        "The supplied operations do not form a group: "
                        f"{a.tolist()} * {b.tolist()} is missing.")

        self._shape = shape
        self._operations: tuple[tuple[float, ...], ...] = tuple(
            _matrix_to_flat(op) for op in ops)
        self._ops = ops
        self._rotations = [op for op in ops if np.linalg.det(op) > 0]
        self._reflections = [op for op in ops if np.linalg.det(op) < 0]
        self._n = len(self._rotations)
        if len(self._reflections) not in (0, self._n):
            raise ValueError(
                f"{self._n} rotations and {len(self._reflections)} reflections "
                "do not form a cyclic or a dihedral group.")

    def __hash__(self):
        return hash((self._shape, self._operations))

    def __eq__(self, other):
        if not isinstance(other, ExplicitPointGroup):
            return NotImplemented
        return (self._shape == other._shape
                and self._operations == other._operations)

    def __iter__(self):
        return iter([op.copy() for op in self._ops])

    # ── irreps ──────────────────────────────────────────────────────────

    def all_irrps(self) -> list[str]:
        '''Labels of the irreps, see the class docstring.'''
        if not self._reflections:
            return [f'j{j}' for j in range(self._n)]
        labels = ['A1', 'A2']
        if self._n % 2 == 0:
            labels += ['B1', 'B2']
        labels += [f'E{j}' for j in range(1, (self._n - 1) // 2 + 1)]
        return labels

    def _rotation_power(self, op: np.ndarray) -> int:
        '''
        Power m (mod n) such that ``op = rotation(2π/n)^m``.
        '''
        angle = np.arctan2(op[1, 0], op[0, 0]) % (2 * np.pi)
        return int(round(angle * self._n / (2 * np.pi))) % self._n

    def character(self, irrep: str, op: np.ndarray) -> complex:
        '''
        Character of *irrep* on *op*.

        Args:
            irrep: one of the labels returned by ``all_irrps``.
            op: an operation matrix of the group.
        Returns:
            The character.
        Raises:
            ValueError: if *irrep* is not a label of this group.
        '''
        arr = np.asarray(op, dtype=float)
        if irrep not in self.all_irrps():
            raise ValueError(f"Unknown irreducible representation: {irrep}")
        if np.linalg.det(arr) > 0:
            m = self._rotation_power(arr)
            if not self._reflections:
                return np.exp(2j * np.pi * int(irrep[1:]) * m / self._n)
            if irrep in ('A1', 'A2'):
                return 1.0
            if irrep in ('B1', 'B2'):
                return 1.0 if m % 2 == 0 else -1.0
            return 2.0 * np.cos(2 * np.pi * int(irrep[1:]) * m / self._n)

        # reflections:  op = sigma_0 r^m  with sigma_0 the first reflection
        m = self._rotation_power(self._reflections[0] @ arr)
        if irrep == 'A1':
            return 1.0
        if irrep == 'A2':
            return -1.0
        if irrep == 'B1':
            return 1.0 if m % 2 == 0 else -1.0
        if irrep == 'B2':
            return -1.0 if m % 2 == 0 else 1.0
        return 0.0
