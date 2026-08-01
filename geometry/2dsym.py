import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import *


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


def _matrix_to_flat(op: np.ndarray) -> tuple[int | float, ...]:
    '''
    Convert a matrix to a flattened tuple suitable for hashing.
    Entries within 1e-10 of an integer are stored as ``int``;
    others are rounded to 12 decimal places and stored as ``float``.
    '''
    flat: list[int | float] = []
    for x in op.flat:
        r = round(x)
        if abs(x - r) < 1e-10:
            flat.append(int(r))
        else:
            flat.append(round(float(x), 12))
    return tuple(flat)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal base class for 2D point groups
# ═══════════════════════════════════════════════════════════════════════════════

class _Generic2DPointGroup(PointGroup):
    '''
    Base class for 2D point groups that store their operations explicitly.

    Operations are stored internally as ``tuple[tuple[int|float,…],…]``
    (flattened data, one tuple per element) together with a single
    ``_shape`` shared by all elements, so the group is hashable.
    Iteration still yields ``np.ndarray`` matrices.
    '''

    def __init__(self, operations: list[np.ndarray]):
        if operations:
            self._shape: tuple[int, ...] = operations[0].shape
        else:
            self._shape = ()
        self._operations: tuple[tuple[int | float, ...], ...] = tuple(
            _matrix_to_flat(op) for op in operations
        )
        self._iter_index = 0

    def __hash__(self):
        return hash((self._shape, self._operations))

    def __eq__(self, other):
        if not isinstance(other, _Generic2DPointGroup):
            return NotImplemented
        return (self._shape == other._shape
                and self._operations == other._operations)

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self) -> np.ndarray:
        if self._iter_index >= len(self._operations):
            raise StopIteration
        data = self._operations[self._iter_index]
        self._iter_index += 1
        return np.array(data, dtype=np.float64).reshape(self._shape)

    # ── fallback irrep / character (for generic subgroups) ───────────────

    def all_irrps(self) -> list[str]:
        return ['A']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        if irrp == 'A':
            return 1.0
        raise ValueError(f"Unknown irreducible representation: {irrp}")

    def little_group(self, k: np.ndarray,
                     reciprocal_basis: np.ndarray | None = None
                     ) -> tuple[PointGroup, list[np.ndarray]]:
        '''
        Fallback little-group decomposition for an unnamed subgroup.

        When *reciprocal_basis* is given, *op* preserves *k* iff
        ``op @ k ≈ k + reciprocal_basis @ n`` for integer *n*.
        Otherwise exact equality ``op @ k ≈ k`` is used.
        '''
        eye = np.eye(k.shape[0]) if k.ndim == 1 else np.eye(2)
        ops = list(self)

        if reciprocal_basis is None:
            if np.allclose(k, 0):
                return self, [eye]
            preserving = [op for op in ops if np.allclose(op @ k, k)]
            others     = [op for op in ops if not np.allclose(op @ k, k)]
        else:
            preserving = [op for op in ops
                          if self._k_preserved(op, k, reciprocal_basis)]
            others     = [op for op in ops
                          if not self._k_preserved(op, k, reciprocal_basis)]

        if len(preserving) <= 1:
            return TrivialGroup(), others[:1] if others else [eye]

        sub = _Generic2DPointGroup(preserving)
        n_coset = (len(ops) // len(preserving)
                   if len(preserving) > 0 else 1)
        return sub, others[:n_coset] if others else [eye]

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_rotation(op: np.ndarray) -> bool:
        '''Return True if *op* is a proper rotation (det > 0).'''
        return np.linalg.det(op) > 0

    @staticmethod
    def _k_preserved(op: np.ndarray, k: np.ndarray,
                     reciprocal_basis: np.ndarray) -> bool:
        '''
        Return True if *op* preserves *k* modulo reciprocal-lattice vectors,
        i.e.  ``op @ k ≈ k + n @ reciprocal_basis`` for some integer *n*.

        *reciprocal_basis* is a (d,d) array whose **rows** are the
        reciprocal basis vectors (same convention as
        ``Model.bz_cellvectors``).
        '''
        delta = op @ k - k
        # Rows of reciprocal_basis are basis vectors b_i.
        # We need integer n such that  n @ reciprocal_basis ≈ delta.
        # Solve: reciprocal_basis.T @ n = delta  (columns = basis vectors).
        n_float = np.linalg.solve(reciprocal_basis.T, delta)
        n_int = np.rint(n_float).astype(int)
        return np.allclose(n_int @ reciprocal_basis, delta)

    def _find_op_index(self, op: np.ndarray) -> int:
        '''Return the index in ``self._operations`` that matches *op*.'''
        for i, g in enumerate(self):
            if np.allclose(g, op):
                return i
        raise ValueError(f"Operation {op} not found in group.")


# ═══════════════════════════════════════════════════════════════════════════════
# Cyclic groups  C₂ · C₃ · C₄ · C₆
# ═══════════════════════════════════════════════════════════════════════════════

class CyclicGroup(_Generic2DPointGroup):
    '''
    Base for cyclic groups Cₙ (order *n*).

    Operations: rotations by 2πk/n for k = 0, 1, …, n−1.
    All irreps are one-dimensional.
    '''

    def __init__(self, n: int):
        self.n = n
        super().__init__([rotation(2 * np.pi * k / n) for k in range(n)])

    def _get_rotation_power(self, op: np.ndarray) -> int:
        '''
        Return the integer *k* (mod *n*) such that *op* ≈ Cₙᵏ (rotation by
        2πk/n).
        '''
        angle = np.arctan2(op[1, 0], op[0, 0])
        angle = angle % (2 * np.pi)
        return int(round(angle * self.n / (2 * np.pi))) % self.n

    # ── little group ─────────────────────────────────────────────────────

    def little_group(self, k: np.ndarray,
                     reciprocal_basis: np.ndarray | None = None
                     ) -> tuple[PointGroup, list[np.ndarray]]:
        '''
        Return the little group of momentum *k*.

        For a cyclic group, the only operations that can leave a non-zero
        momentum invariant are the identity (and C₂ when *k* is at a
        time-reversal-invariant momentum point).
        '''
        if reciprocal_basis is None and np.allclose(k, 0):
            return self, [np.eye(2)]

        ops = list(self)

        if reciprocal_basis is None:
            preserving = [op for op in ops if np.allclose(op @ k, k)]
            others     = [op for op in ops if not np.allclose(op @ k, k)]
        else:
            preserving = [op for op in ops
                          if self._k_preserved(op, k, reciprocal_basis)]
            others     = [op for op in ops
                          if not self._k_preserved(op, k, reciprocal_basis)]

        m = len(preserving)

        if m == 1:
            return TrivialGroup(), others[:1] if others else [np.eye(2)]
        if m == self.n:
            return self, [np.eye(2)]

        # Build a generic subgroup with the preserving operations
        sub = _Generic2DPointGroup(preserving)
        return sub, others[:self.n // m] if others else [np.eye(2)]


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
        super().__init__(2)

    def all_irrps(self) -> list[str]:
        return ['A', 'B']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        k = self._get_rotation_power(op)          # 0 → E,  1 → C₂
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
        super().__init__(3)

    def all_irrps(self) -> list[str]:
        return ['A', 'E1', 'E2']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        k = self._get_rotation_power(op)          # 0 → E, 1 → C₃, 2 → C₃²
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
        super().__init__(4)

    def all_irrps(self) -> list[str]:
        return ['A', 'B', 'E1', 'E2']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        k = self._get_rotation_power(op)   # 0→E, 1→C₄, 2→C₂, 3→C₄³
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
    E₁      1    ω    ω²  −1   ω⁴     ω⁵
    E₂      1   ω²    ω⁴   1   ω²     ω⁴
    E₁*     1   ω⁵    ω⁴  −1   ω²     ω
    E₂*     1   ω⁴    ω²   1   ω⁴     ω²
    =====  ===  ====  ===  ===  =====  =====

    ω = exp(πi/3).
    '''

    def __init__(self):
        super().__init__(6)

    def all_irrps(self) -> list[str]:
        return ['A', 'B', 'E1', 'E2', 'E1*', 'E2*']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        k = self._get_rotation_power(op)   # 0→E, 1→C₆, 2→C₃, 3→C₂, 4→C₃², 5→C₆⁵
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

class DihedralGroup(_Generic2DPointGroup):
    '''
    Base for dihedral groups Dₙ (order 2*n*).

    Operations
    ----------
    - *n* rotations Cₙᵏ by 2πk/n  (k = 0, …, n−1)
    - *n* reflections σ about axes at angles πk/n  (k = 0, …, n−1)

    Reflections are stored **after** rotations in ``_operations``.
    '''

    def __init__(self, n: int):
        self.n = n
        # Rotations
        rots = [rotation(2 * np.pi * k / n) for k in range(n)]
        # Reflections: axis at angle π·k / n
        refs = [reflection(np.pi * k / n) for k in range(n)]
        super().__init__(rots + refs)

    # ── identifying operations ───────────────────────────────────────────

    def _get_rotation_power(self, op: np.ndarray) -> int:
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
            k = self._get_rotation_power(op)
            if 2 * k == self.n:                 # rotation by π
                return 'C2'
            return 'Cn'                         # paired ± rotation

        # Reflection
        if self.n % 2 == 1:
            return 'sigma'                      # all reflections in one class
        k = self._get_rotation_power(op)
        return 'sv' if k % 2 == 0 else 'sd'

    # ── little group ─────────────────────────────────────────────────────

    def little_group(self, k: np.ndarray,
                     reciprocal_basis: np.ndarray | None = None
                     ) -> tuple[PointGroup, list[np.ndarray]]:
        '''
        Return the little group of momentum *k*.

        For a dihedral group, non-zero *k* may be preserved by:
        - the identity (always)
        - a reflection whose axis is perpendicular to *k*
        - C₂ (rotation by π) when *k* is at a TRIM point
        '''
        if reciprocal_basis is None and np.allclose(k, 0):
            return self, [np.eye(2)]

        ops = list(self)

        if reciprocal_basis is None:
            preserving = [op for op in ops if np.allclose(op @ k, k)]
            others     = [op for op in ops if not np.allclose(op @ k, k)]
        else:
            preserving = [op for op in ops
                          if self._k_preserved(op, k, reciprocal_basis)]
            others     = [op for op in ops
                          if not self._k_preserved(op, k, reciprocal_basis)]

        m = len(preserving)

        if m == 1:
            return TrivialGroup(), others[:1] if others else [np.eye(2)]
        if m == len(ops):
            return self, [np.eye(2)]

        sub = _Generic2DPointGroup(preserving)
        return sub, others[:len(ops) // m] if others else [np.eye(2)]


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
        super().__init__(3)

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
                    # reflections
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
        super().__init__(4)

    def all_irrps(self) -> list[str]:
        return ['A1', 'A2', 'B1', 'B2', 'E']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        is_rot = self._is_rotation(op)
        k = self._get_rotation_power(op)       # rotation power / axis index (see docstring)
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
        super().__init__(6)

    def all_irrps(self) -> list[str]:
        return ['A1', 'A2', 'B1', 'B2', 'E1', 'E2']

    def character(self, irrp: str, op: np.ndarray) -> complex:
        is_rot = self._is_rotation(op)
        k = self._get_rotation_power(op)       # rotation power / axis index

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
