import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import representation
from base import PointGroup


def rotation(rad: float) -> np.ndarray:
    '''
    Create a 2D rotation matrix for a given angle in radians.
    Args:
        rad: The angle in radians to rotate.
    Returns:
        A 2x2 numpy array representing the rotation matrix.
    '''
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s],
                     [s,  c]])


def reflection(rad: float) -> np.ndarray:
    '''
    Create a 2D reflection matrix about an axis at angle `rad` from the x-axis.
    The reflection maps (x,y) to its mirror image across the line through the
    origin with direction angle `rad`.

    Args:
        rad: The angle (in radians) of the reflection axis from the x-axis.
    Returns:
        A 2x2 numpy array representing the reflection matrix (det = -1).
    '''
    c, s = np.cos(2 * rad), np.sin(2 * rad)
    return np.array([[c,  s],
                     [s, -c]])


# ═══════════════════════════════════════════════════════════════════════════════
# Cyclic groups  C₂ · C₃ · C₄ · C₆
# ═══════════════════════════════════════════════════════════════════════════════

def cyclic_operations(n: int) -> list[np.ndarray]:
    '''
    The *n* rotations of C*n*.

    Args:
        n: the order of the rotation subgroup.
    Returns:
        The rotations by 2πk/n for k = 0, …, n−1.
    '''
    return [rotation(2 * np.pi * k / n) for k in range(n)]


def dihedral_operations(n: int) -> list[np.ndarray]:
    '''
    The *n* rotations and *n* reflections of D*n*.

    Args:
        n: the order of the rotation subgroup.
    Returns:
        The rotations by 2πk/n, then the reflections about the axes at πk/n.
    '''
    return cyclic_operations(n) + [reflection(np.pi * k / n) for k in range(n)]


class CyclicGroup(PointGroup):
    '''
    A cyclic group C*n*: *n* rotations and nothing else.

    Its irreps are *n* one-dimensional ones, ``'j0' … 'j{n-1}'``, with
    ``χ_j(r^m) = exp(2πi j m / n)``.
    '''

    def __init__(self, operations: list[np.ndarray]):
        '''
        Args:
            operations: the rotations of the group, as matrices.
        '''
        self._ops = [np.asarray(op, dtype=float) for op in operations]
        self.n = len(self._ops)

    def __iter__(self):
        return iter([op.copy() for op in self._ops])

    def __eq__(self, other):
        if not isinstance(other, CyclicGroup):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def _key(self) -> tuple:
        '''The operations as a hashable value, for comparison.'''
        return tuple(representation._matrix_to_flat(op) for op in self._ops)

    @property
    def order(self) -> int:
        '''The number of operations, *n*.'''
        return self.n

    def _rotation_power(self, op: np.ndarray) -> int:
        '''
        The power *m* (mod *n*) such that *op* is the rotation by 2πm/n.
        '''
        angle = np.arctan2(op[1, 0], op[0, 0]) % (2 * np.pi)
        return int(round(angle * self.n / (2 * np.pi))) % self.n

    def all_irrps(self) -> list[str]:
        return [f'j{j}' for j in range(self.n)]

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
        return np.exp(2j * np.pi * int(irrep[1:]) * self._rotation_power(op) / self.n)


class DihedralGroup(PointGroup):
    '''
    A dihedral group D*n*: *n* rotations and *n* reflections.

    Its irreps are ``'A1'``, ``'A2'`` (plus ``'B1'``, ``'B2'`` for even *n*) and the
    two-dimensional ``'E1' …``, with ``χ_{E_j}(r^m) = 2 cos(2π j m / n)`` on rotations
    and 0 on reflections.  A reflection is written ``σ = σ_0 r^m`` with σ_0 the first
    reflection of the list; its character is ``(-1)^m`` for ``B1`` and ``-(-1)^m`` for
    ``B2``.
    '''

    def __init__(self, operations: list[np.ndarray]):
        '''
        Args:
            operations: the rotations and reflections of the group, as matrices.
        '''
        self._ops = [np.asarray(op, dtype=float) for op in operations]
        self._reflections = [op for op in self._ops if np.linalg.det(op) < 0]
        self.n = len(self._ops) - len(self._reflections)

    def __iter__(self):
        return iter([op.copy() for op in self._ops])

    def __eq__(self, other):
        if not isinstance(other, DihedralGroup):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def _key(self) -> tuple:
        '''The operations as a hashable value, for comparison.'''
        return tuple(representation._matrix_to_flat(op) for op in self._ops)

    @property
    def order(self) -> int:
        '''The number of operations, *2n*.'''
        return len(self._ops)

    @staticmethod
    def _is_rotation(op: np.ndarray) -> bool:
        '''Return True if *op* is a proper rotation (det > 0).'''
        return np.linalg.det(np.asarray(op, dtype=float)) > 0

    def _rotation_power(self, op: np.ndarray) -> int:
        '''
        The power *m* (mod *n*) such that *op* is the rotation by 2πm/n.
        '''
        angle = np.arctan2(op[1, 0], op[0, 0]) % (2 * np.pi)
        return int(round(angle * self.n / (2 * np.pi))) % self.n

    def all_irrps(self) -> list[str]:
        labels = ['A1', 'A2']
        if self.n % 2 == 0:
            labels += ['B1', 'B2']
        return labels + [f'E{j}' for j in range(1, (self.n - 1) // 2 + 1)]

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
            return 2.0 * np.cos(2 * np.pi * int(irrep[1:]) * m / self.n)
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

    def _rotation_power(self, op: np.ndarray) -> int:
        '''
        If *op* is a **rotation**, return the power *k* (mod *n*) such that
        *op* ≈ Cₙᵏ.

        If *op* is a **reflection**, return the axis index *k* (mod *n*)
        such that the reflection axis is at angle πk/n.
        '''
        if self._is_rotation(op):
            angle = np.arctan2(op[1, 0], op[0, 0])
            angle = angle % (2 * np.pi)
            return int(round(angle * self.n / (2 * np.pi))) % self.n
        else:
            # Reflection matrix: [[cos(2θ), sin(2θ)], [sin(2θ), -cos(2θ)]]
            # where θ is the axis angle.
            two_theta = np.arctan2(op[1, 0], op[0, 0])
            theta = (two_theta / 2) % np.pi
            return int(round(theta * self.n / np.pi)) % self.n

    def _conjugacy_class(self, op: np.ndarray) -> str:
        '''
        Return a label for the conjugacy class of *op*.

        ===========  ================================
        Label        Meaning
        ===========  ================================
        ``'E'``      identity
        ``'C2'``     rotation by π (only when *n* even)
        ``'Cn'``     paired rotations ±2πk/n
        ``'sigma'``  all reflections (when *n* odd)
        ``'sv'``     reflections through vertices (*n* even, axis index even)
        ``'sd'``     reflections through edges (*n* even, axis index odd)
        ===========  ================================
        '''
        if np.allclose(op, np.eye(2)):
            return 'E'

        if self._is_rotation(op):
            k = self._rotation_power(op)
            if 2 * k == self.n:                 # rotation by π
                return 'C2'
            return 'Cn'                         # paired ± rotation

        # Reflection
        if self.n % 2 == 1:
            return 'sigma'                      # all reflections in one class
        k = self._rotation_power(op)
        return 'sv' if k % 2 == 0 else 'sd'


def cyclic_of_order(n: int) -> CyclicGroup:
    '''
    The cyclic group of order *n*.

    Args:
        n: the order.
    Returns:
        C*n*.
    '''
    return CyclicGroup(cyclic_operations(n))


def dihedral_of_order(n: int) -> DihedralGroup:
    '''
    The dihedral group of order *2n*.

    Args:
        n: the order of its rotation subgroup.
    Returns:
        D*n*.
    '''
    return DihedralGroup(dihedral_operations(n))


def group_of(operations: list[np.ndarray]) -> PointGroup:
    '''
    The group a list of operations is, as the kind it turns out to be.

    Args:
        operations: the operations, as matrices.
    Returns:
        A :class:`CyclicGroup` or a :class:`DihedralGroup`.
    '''
    reflections = sum(1 for op in operations if np.linalg.det(np.asarray(op, dtype=float)) < 0)
    return CyclicGroup(operations) if reflections == 0 else DihedralGroup(operations)


# ── C₂ ──────────────────────────────────────────────────────────────────────

class C2(CyclicGroup):
    '''
    Cyclic group of order 2: {E, C₂}.

    Irreps
    ------
    =====  ===  ====
    Irrep   E   C₂
    =====  ===  ====
    A       1    1
    B       1   −1
    =====  ===  ====
    '''

    def __init__(self):
        super().__init__(cyclic_operations(2))

    def all_irrps(self) -> list[str]:
        return ['A', 'B']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        k = self._rotation_power(op)          # 0 → E,  1 → C₂
        if irrp == 'A':
            return 1.0
        if irrp == 'B':
            return 1.0 if k == 0 else -1.0
        raise ValueError(f"Unknown irreducible representation: {irrp}")


# ── C₃ ──────────────────────────────────────────────────────────────────────

class C3(CyclicGroup):
    '''
    Cyclic group of order 3: {E, C₃, C₃²}.

    Irreps
    ------
    =====  ===  =====  ======
    Irrep   E    C₃     C₃²
    =====  ===  =====  ======
    A       1    1      1
    E₁      1    ω      ω²
    E₂      1    ω²     ω
    =====  ===  =====  ======

    ω = exp(2πi/3).
    '''

    def __init__(self):
        super().__init__(cyclic_operations(3))

    def all_irrps(self) -> list[str]:
        return ['A', 'E1', 'E2']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        k = self._rotation_power(op)          # 0 → E, 1 → C₃, 2 → C₃²
        omega = np.exp(2j * np.pi / 3)
        match irrp:
            case 'A':
                return 1.0
            case 'E1':
                return omega ** k
            case 'E2':
                return omega ** (2 * k)
        raise ValueError(f"Unknown irreducible representation: {irrp}")


# ── C₄ ──────────────────────────────────────────────────────────────────────

class C4(CyclicGroup):
    '''
    Cyclic group of order 4: {E, C₄, C₂, C₄³}.

    Irreps
    ------
    =====  ===  ====  ===  =====
    Irrep   E    C₄    C₂   C₄³
    =====  ===  ====  ===  =====
    A       1    1     1    1
    B       1   −1     1   −1
    E₁      1    i    −1   −i
    E₂      1   −i    −1    i
    =====  ===  ====  ===  =====
    '''

    def __init__(self):
        super().__init__(cyclic_operations(4))

    def all_irrps(self) -> list[str]:
        return ['A', 'B', 'E1', 'E2']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        k = self._rotation_power(op)   # 0→E, 1→C₄, 2→C₂, 3→C₄³
        match irrp:
            case 'A':
                return 1.0
            case 'B':
                return 1.0 if k % 2 == 0 else -1.0
            case 'E1':
                return 1j ** k
            case 'E2':
                return (-1j) ** k
        raise ValueError(f"Unknown irreducible representation: {irrp}")


# ── C₆ ──────────────────────────────────────────────────────────────────────

class C6(CyclicGroup):
    '''
    Cyclic group of order 6: {E, C₆, C₃, C₂, C₃², C₆⁵}.

    Irreps
    ------
    =====  ===  ====  ===  ===  =====  =====
    Irrep   E    C₆    C₃   C₂   C₃²    C₆⁵
    =====  ===  ====  ===  ===  =====  =====
    A       1    1     1    1    1      1
    B       1   −1     1   −1    1     −1
    E₁      1    ω    ω²   −1   ω⁴     ω⁵
    E₂      1   ω²    ω⁴   1    ω²     ω⁴
    E₁*     1   ω⁵    ω⁴  −1    ω²     ω
    E₂*     1   ω⁴    ω²   1    ω⁴     ω²
    =====  ===  ====  ===  ===  =====  =====

    ω = exp(πi/3).
    '''

    def __init__(self):
        super().__init__(cyclic_operations(6))

    def all_irrps(self) -> list[str]:
        return ['A', 'B', 'E1', 'E2', 'E1*', 'E2*']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        k = self._rotation_power(op)   # 0→E, 1→C₆, 2→C₃, 3→C₂, 4→C₃², 5→C₆⁵
        omega = np.exp(1j * np.pi / 3)     # exp(πi/3)

        match irrp:
            case 'A':
                return 1.0
            case 'B':
                return 1.0 if k % 2 == 0 else -1.0
            case 'E1':
                return omega ** k
            case 'E2':
                return omega ** (2 * k)
            case 'E1*':
                return omega ** (5 * k)
            case 'E2*':
                return omega ** (4 * k)
        raise ValueError(f"Unknown irreducible representation: {irrp}")


# ═══════════════════════════════════════════════════════════════════════════════
# Dihedral groups  D₃ · D₄ · D₆
# ═══════════════════════════════════════════════════════════════════════════════

# ── D₃ ──────────────────────────────────────────────────────────────────────

class D3(DihedralGroup):
    '''
    Dihedral group of order 6: {E, C₃, C₃², 3×σ}.

    Conjugacy classes: {E}, {C₃, C₃²}, {all 3 reflections}.

    =====  ===  ======  =======
    Irrep   E   2 C₃    3 σ
    =====  ===  ======  =======
    A₁      1    1       1
    A₂      1    1      −1
    E       2   −1       0
    =====  ===  ======  =======
    '''

    def __init__(self):
        super().__init__(dihedral_operations(3))

    def all_irrps(self) -> list[str]:
        return ['A1', 'A2', 'E']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        cls = self._conjugacy_class(op)
        is_rot = self._is_rotation(op)

        match irrp:
            case 'A1':
                return 1.0
            case 'A2':
                return 1.0 if is_rot else -1.0
            case 'E':
                if cls == 'E':
                    return 2.0
                if cls == 'Cn':
                    return -1.0
                return 0.0                      # reflections
        raise ValueError(f"Unknown irreducible representation: {irrp}")


# ── D₄ ──────────────────────────────────────────────────────────────────────

class D4(DihedralGroup):
    '''
    Dihedral group of order 8: {E, C₄, C₂, C₄³, σᵥ, σᵥ′, σ_d, σ_d′}.

    Conjugacy classes: {E}, {C₂}, {C₄, C₄³}, {σᵥ, σᵥ′}, {σ_d, σ_d′}.

    =====  ===  ===  ======  =======  =======
    Irrep   E   C₂   2 C₄    2 σᵥ     2 σ_d
    =====  ===  ===  ======  =======  =======
    A₁      1    1    1       1        1
    A₂      1    1    1      −1       −1
    B₁      1    1   −1       1       −1
    B₂      1    1   −1      −1        1
    E       2   −2    0       0        0
    =====  ===  ===  ======  =======  =======
    '''

    def __init__(self):
        super().__init__(dihedral_operations(4))

    def all_irrps(self) -> list[str]:
        return ['A1', 'A2', 'B1', 'B2', 'E']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        is_rot = self._is_rotation(op)
        k = self._rotation_power(op)       # rotation power / axis index (see docstring)
        cls = self._conjugacy_class(op)
        match irrp:
            case 'A1':
                return 1.0
            case 'A2':
                return 1.0 if is_rot else -1.0
            case 'B1':
                # +1 on even powers / vertex reflections; −1 on odd powers / edge reflections
                return 1.0 if k % 2 == 0 else -1.0
            case 'B2':
                if is_rot:
                    return 1.0 if k % 2 == 0 else -1.0
                else:
                    return -1.0 if k % 2 == 0 else 1.0   # swapped relative to B₁ for reflections
            case 'E':
                if cls == 'E':
                    return 2.0
                if cls == 'C2':
                    return -2.0
                return 0.0                       # C₄ or any reflection
        raise ValueError(f"Unknown irreducible representation: {irrp}")

# ── D₆ ──────────────────────────────────────────────────────────────────────

class D6(DihedralGroup):
    '''
    Dihedral group of order 12: 6 rotations + 6 reflections.

    Conjugacy classes: {E}, {C₂}, {C₃, C₃²}, {C₆, C₆⁵},
    {3 σᵥ through vertices}, {3 σ_d through edges}.

    =====  ===  ===  ======  ======  =====  =====
    Irrep   E   C₂   2 C₃   2 C₆   3 σᵥ   3 σ_d
    =====  ===  ===  ======  ======  =====  =====
    A₁      1    1    1       1      1      1
    A₂      1    1    1       1     −1     −1
    B₁      1   −1    1      −1      1     −1
    B₂      1   −1    1      −1     −1      1
    E₁      2   −2   −1       1      0      0
    E₂      2    2   −1      −1      0      0
    =====  ===  ===  ======  ======  =====  =====
    '''

    def __init__(self):
        super().__init__(dihedral_operations(6))

    def all_irrps(self) -> list[str]:
        return ['A1', 'A2', 'B1', 'B2', 'E1', 'E2']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        is_rot = self._is_rotation(op)
        k = self._rotation_power(op)       # rotation power / axis index

        match irrp:
            case 'A1':
                return 1.0
            case 'A2':
                return 1.0 if is_rot else -1.0
            case 'B1':
                # +1 for even k, −1 for odd k  (both rotations and reflections)
                return 1.0 if k % 2 == 0 else -1.0
            case 'B2':
                if is_rot:
                    return 1.0 if k % 2 == 0 else -1.0
                else:
                    return -1.0 if k % 2 == 0 else 1.0
            case 'E1':
                if is_rot:
                    return 2.0 * np.cos(2 * np.pi * k / self.n)
                return 0.0
            case 'E2':
                if is_rot:
                    return 2.0 * np.cos(4 * np.pi * k / self.n)
                return 0.0

        raise ValueError(f"Unknown irreducible representation: {irrp}")
