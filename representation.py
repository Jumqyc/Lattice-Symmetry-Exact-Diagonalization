'''
Representation theory: the character tables.

This module answers one question only -- *which irreducible representations
does this group have, and what are their characters*.  It says nothing about
the lattice and nothing about the Hilbert space.

Two kinds of group live here:

* :class:`TrivialGroup`, the group with one element;
* :func:`group_of`, which builds the cyclic or dihedral group a list of operations
  is, and :class:`CyclicGroup` / :class:`DihedralGroup`, which know their own
  character tables.  This is what the little groups are built from, since a little
  group is a generic subgroup and has no dedicated class.

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
    A point group given by its list of operation matrices, of one of the two kinds
    a finite subgroup of O(2) can be.

    This class holds only what such a list has in common: it must be orthogonal and
    closed, and it must be cyclic or dihedral.  Which kind it is, is decided on
    construction, and the table is left to the kind -- :class:`CyclicGroup` and
    :class:`DihedralGroup` write their own, because the two have nothing in common
    but the construction.  A list that is neither is refused here rather than given
    a table it does not have.

    Use :func:`group_of` to build one from a list.
    '''

    def __init__(self, operations: list[np.ndarray]):
        ops = _unique_ops(operations)
        if not ops:
            raise ValueError("A point group needs at least one operation.")
        shape = ops[0].shape
        if shape != (2, 2):
            raise ValueError(
                "This module knows the finite subgroups of O(2) only -- cyclic and "
                f"dihedral -- and got operations of shape {shape}.")
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
                "are neither cyclic nor dihedral.")

    @property
    def order(self) -> int:
        '''The number of operations, ``n`` for a cyclic group and ``2n`` for a
        dihedral one.'''
        return len(self._ops)

    def __hash__(self):
        return hash((self._shape, self._operations))

    def __eq__(self, other):
        if not isinstance(other, ExplicitPointGroup):
            return NotImplemented
        return (self._shape == other._shape
                and self._operations == other._operations)

    def __iter__(self):
        return iter([op.copy() for op in self._ops])

    # ── operations ──────────────────────────────────────────────────────

    @staticmethod
    def _is_rotation(op: np.ndarray) -> bool:
        '''Return True if *op* is a proper rotation (det > 0).'''
        return np.linalg.det(np.asarray(op, dtype=float)) > 0

    def _rotation_power(self, op: np.ndarray) -> int:
        '''
        The power *m* (mod *n*) such that *op* is the rotation by 2πm/n.

        Args:
            op: a rotation of the group.
        Returns:
            The power, as an integer in ``[0, n)``.
        '''
        angle = np.arctan2(op[1, 0], op[0, 0]) % (2 * np.pi)
        return int(round(angle * self._n / (2 * np.pi))) % self._n

    def all_irrps(self) -> list[str]:
        raise NotImplementedError("the kind of group provides the table")

    def character(self, irrep: str, op: np.ndarray) -> complex:
        raise NotImplementedError("the kind of group provides the table")


class CyclicGroup(ExplicitPointGroup):
    '''
    A cyclic group C*n*: *n* rotations, no reflection.

    Its irreps are *n* one-dimensional ones, ``'j0' … 'j{n-1}'``, with
    ``χ_j(r^m) = exp(2πi j m / n)``.
    '''

    def all_irrps(self) -> list[str]:
        return [f'j{j}' for j in range(self._n)]

    def character(self, irrep: str, op: np.ndarray) -> complex:
        '''
        Character of *irrep* on *op*.

        Args:
            irrep: one of the labels returned by ``all_irrps``.
            op: an operation of the group.
        Returns:
            The character.
        Raises:
            ValueError: if *irrep* is not a label of this group.
        '''
        if irrep not in self.all_irrps():
            raise ValueError(f"Unknown irreducible representation: {irrep}")
        return np.exp(2j * np.pi * int(irrep[1:]) * self._rotation_power(op) / self._n)


class DihedralGroup(ExplicitPointGroup):
    '''
    A dihedral group D*n*: *n* rotations and *n* reflections.

    Its irreps are ``'A1'``, ``'A2'`` (plus ``'B1'``, ``'B2'`` for even *n*) and the
    two-dimensional ``'E1' …``, with ``χ_{E_j}(r^m) = 2 cos(2π j m / n)`` on rotations
    and 0 on reflections.  A reflection is written ``σ = σ_0 r^m`` with σ_0 the first
    reflection of the list; its character is ``(-1)^m`` for ``B1`` and ``-(-1)^m`` for
    ``B2``.
    '''

    def all_irrps(self) -> list[str]:
        labels = ['A1', 'A2']
        if self._n % 2 == 0:
            labels += ['B1', 'B2']
        return labels + [f'E{j}' for j in range(1, (self._n - 1) // 2 + 1)]

    def character(self, irrep: str, op: np.ndarray) -> complex:
        '''
        Character of *irrep* on *op*.

        Args:
            irrep: one of the labels returned by ``all_irrps``.
            op: an operation of the group.
        Returns:
            The character.
        Raises:
            ValueError: if *irrep* is not a label of this group.
        '''
        if irrep not in self.all_irrps():
            raise ValueError(f"Unknown irreducible representation: {irrep}")
        arr = np.asarray(op, dtype=float)
        if self._is_rotation(arr):
            m = self._rotation_power(arr)
            if irrep in ('A1', 'A2'):
                return 1.0
            if irrep in ('B1', 'B2'):
                return 1.0 if m % 2 == 0 else -1.0
            return 2.0 * np.cos(2 * np.pi * int(irrep[1:]) * m / self._n)
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


def group_of(operations: list[np.ndarray]) -> ExplicitPointGroup:
    '''
    The group a list of operations is, as the kind it turns out to be.

    Args:
        operations: the operations, as matrices.
    Returns:
        A :class:`CyclicGroup` or a :class:`DihedralGroup`.
    Raises:
        ValueError: if the list is not a group, is not orthogonal, is not
            two-dimensional, or is neither cyclic nor dihedral.
    '''
    probe = ExplicitPointGroup(operations)      # validates and classifies
    kind = CyclicGroup if not probe._reflections else DihedralGroup
    return kind(operations)
