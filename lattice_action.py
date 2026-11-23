'''
Geometry: how the symmetry group acts on the lattice.

This module answers the questions that need coordinates rather than
characters:

* what are the translations of the supercell, and what momentum labels do
  they carry (:class:`TranslationGroup`);
* how does one operation move the momentum labels, and which operations
  leave a given label fixed (the little group);
* which operations are symmetries of the finite lattice at all.

Nothing here knows about the Hilbert space.  The lattice rules are functions of
an operation matrix and of the supercell basis, which is why they are free
functions rather than methods of the group: the group is representation theory
and must not depend on geometry.
'''

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from base import LatticeAction, PointGroup
from representation import (ExplicitPointGroup, TrivialGroup, _right_coset_representatives,
                            _unique_ops)


# positions are compared after rounding to this many decimals; a symmetry
# operation maps a site onto another site only up to a supercell-lattice vector
_POS_TOL_DIGITS = 8
# a position is on a lattice site if its coordinates are integral to this
# tolerance (in units of the supercell vectors)
_POS_TOL = 1e-6


class TranslationGroup:
    '''
    Translation group  G_tr = Z_{N1} × … × Z_{Nd}.

    Elements are integer tuples  t = (t₁, …, t_d)  with  0 ≤ tᵢ < Nᵢ.
    Irreducible representations are labelled by momentum tuples
    k = (k₁, …, k_d)  in the Brillouin zone.

    Characters are purely combinatorial:
        χ_k(t) = exp(2πi · Σ_j k_j t_j / N_j).
    '''
    def __init__(self, Nvec: tuple[int, ...],
                 cell_vectors: np.ndarray):
        self.Nvec = Nvec
        self.dim = len(Nvec)
        self.cell_vectors = cell_vectors

    def __len__(self) -> int:
        return int(np.prod(self.Nvec))

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        for ns in product(*[range(Ni) for Ni in self.Nvec]):
            yield ns

    def translation_vector(self, t: tuple[int, ...]) -> np.ndarray:
        '''Real-space translation vector  Σ tᵢ aᵢ.'''
        return sum(ti * self.cell_vectors[i] for i, ti in enumerate(t))

    def all_momenta(self) -> list[tuple[int, ...]]:
        '''All momentum labels in the Brillouin zone.'''
        return list(self)

    def momentum_vector(self, k: tuple[int, ...],
                        reciprocal_basis: np.ndarray) -> np.ndarray:
        '''Real-space momentum vector  k = Σ (kᵢ/Nᵢ) bᵢ.'''
        return np.array(k) @ reciprocal_basis

    def character(self, k: tuple[int, ...],
                  t: tuple[int, ...]) -> complex:
        '''χ_k(t) = exp(2πi · Σ_j k_j t_j / N_j).'''
        phase = 2.0 * np.pi * sum(kj * tj / Nj
                                   for kj, tj, Nj in zip(k, t, self.Nvec))
        return np.exp(1j * phase)


def label_action(op: np.ndarray, supercell_vectors: np.ndarray) -> np.ndarray:
    '''
    Integer action of *op* on the momentum labels.

    With ``A_i`` the supercell vectors (rows of *supercell_vectors*), an
    operation acts on the lattice as ``op A_i = Σ_j M[j, i] A_j`` and
    therefore maps the momentum label k (an integer tuple, in units of the
    supercell reciprocal vectors) to ``Mᵀ k``.

    Args:
        op: the operation matrix.
        supercell_vectors: ``(d, d)`` array whose *rows* are the supercell
            vectors.
    Returns:
        The integer matrix M.
    Raises:
        ValueError: if *op* does not preserve the supercell lattice, in
            which case it is not a symmetry of the finite lattice and the
            momentum labels are not defined.
    '''
    arr = np.asarray(op, dtype=float)
    cell = np.asarray(supercell_vectors, dtype=float)
    if arr.ndim == 0:
        return np.eye(cell.shape[0], dtype=int)
    M = ((arr @ cell.T).T @ np.linalg.inv(cell)).T
    if not np.allclose(M, np.rint(M), atol=1e-8):
        raise ValueError(
            f"Operation {arr.tolist()} does not preserve the supercell lattice "
            f"spanned by {cell.tolist()}: it is not a symmetry of this lattice.")
    return np.rint(M).astype(int)


def preserves_momentum(op: np.ndarray, k: tuple[int, ...],
                       supercell_vectors: np.ndarray,
                       Nvec: np.ndarray) -> bool:
    '''
    Whether *op* leaves the momentum label *k* invariant.

    The labels live in ``Z_{N_1} x … x Z_{N_d}``, so the condition is
    ``Mᵀ k ≡ k (mod N_i)``.  It is *not* enough to check ``op k ≡ k`` modulo
    the reciprocal lattice: every label already is an integer combination of
    the supercell reciprocal vectors, so that test accepts every operation,
    which makes the little group equal to the whole point group at every
    momentum and destroys the point-group symmetry labels.

    Args:
        op: the operation matrix.
        k: the momentum label.
        supercell_vectors: rows are the supercell vectors.
        Nvec: multiplicities ``N_i`` of the unit cell vectors.
    Returns:
        True if *op* belongs to the little group of *k*.
    '''
    M = label_action(op, supercell_vectors)
    label = np.asarray(k, dtype=int)
    return bool(np.all((M.T @ label - label) % np.asarray(Nvec, dtype=int) == 0))


def little_group(point_group: PointGroup, k: tuple[int, ...],
                 supercell_vectors: np.ndarray, Nvec: np.ndarray
                 ) -> tuple[PointGroup, list[np.ndarray]]:
    '''
    Little group of the momentum label *k*, plus coset representatives.

    Args:
        point_group: the full point group.
        k: momentum label, an integer tuple.
        supercell_vectors: rows are the supercell vectors.
        Nvec: multiplicities of the unit cell vectors.
    Returns:
        (G_k, coset_reps): the subgroup of operations that leave *k*
        invariant and a list of representatives of the right cosets
        ``G_k \\ G_pt``.
    '''
    ops = _unique_ops(list(point_group))
    preserving = [op for op in ops
                  if preserves_momentum(op, k, supercell_vectors, Nvec)]
    coset_reps = _right_coset_representatives(ops, preserving)
    if len(preserving) == len(ops):
        group: PointGroup = point_group
    elif len(preserving) == 1:
        group = TrivialGroup()
    else:
        group = ExplicitPointGroup(preserving)
    return group, coset_reps


class Lattice:
    '''
    Geometry of the finite lattice: the supercell, its sites and an exact
    position lookup.

    Sites are addressed by their type and by their position modulo the
    supercell lattice, so a symmetry operation may land on any periodic image
    of a site and still be recognised.
    '''

    def __init__(self,
                 supercell_info: list[tuple[np.ndarray, int]],
                 cell_info: dict[tuple[str, int], np.ndarray],
                 MAX_EXTENSION: int = 1):
        '''
        Args:
            supercell_info: one ``(vector, multiplicity)`` pair per lattice
                direction; the supercell vectors are ``vector * multiplicity``.
            cell_info: position of every site inside the cell, keyed by
                ``(species, label)``.  The species is the equivalence class a
                symmetry operation has to preserve -- two sites of the same
                species may be exchanged (the two sublattices of a honeycomb,
                say), two sites of different species may not.  The label only
                has to make the key unique.
            MAX_EXTENSION: half-width of the periodically extended copy, kept
                for plotting only.
        Raises:
            TypeError: if a supercell vector is not a numpy array.
            ValueError: if the vectors have inconsistent dimensions, if a
                multiplicity is not a positive integer, or if two sites of the
                same type share a position.
        '''
        self.spatial_dim = len(supercell_info[0])

        if not all(isinstance(vec, np.ndarray) for vec, _ in supercell_info):
            raise TypeError("All supercell vectors must be numpy arrays.")
        if not all(len(vec) == self.spatial_dim for vec, _ in supercell_info):
            raise ValueError("All supercell vectors must have the same dimension {}."
                             .format(self.spatial_dim))
        if not all(isinstance(m, int) and m > 0 for _, m in supercell_info):
            raise ValueError("All multiplicities must be positive integers.")

        self.cell_vectors: NDArray[np.float64] = np.array([vec for vec, _ in supercell_info])
        self.Nvec: NDArray[np.int64] = np.array([m for _, m in supercell_info])
        self.supercell_vectors: NDArray[np.float64] = self.cell_vectors * self.Nvec[:, np.newaxis]
        if not self.cell_vectors.shape == (self.spatial_dim, self.spatial_dim):
            raise ValueError("The number of supercell vectors must match the spatial dimension.")
        self.bz_cellvectors = 2 * np.pi * np.linalg.inv(self.supercell_vectors).T

        self.unit_cell_idx: list[tuple[np.ndarray, str, int]] = []
        for n in product(*[range(m) for _, m in supercell_info]):
            for (s, _), vec in cell_info.items():
                # centering the cell at the origin and giving every atom a unique index
                self.unit_cell_idx.append(
                    (vec + np.array(n) @ self.cell_vectors, s, len(self.unit_cell_idx)))
        self.size_of_unit_cell = len(self.unit_cell_idx)

        self._site_lookup: dict[tuple[str, tuple[float, ...]], int] = {}
        for pos, s, idx in self.unit_cell_idx:
            key = (s, self.position_key(pos))
            if key in self._site_lookup:
                raise ValueError(f"Two sites of type {s!r} occupy the same position {pos}.")
            self._site_lookup[key] = idx

        # periodically extended copy, used for plotting only
        self.extended_cell = []
        for pos, s, idx in self.unit_cell_idx:
            for n in product(range(-MAX_EXTENSION, MAX_EXTENSION + 1),
                             repeat=self.spatial_dim):
                self.extended_cell.append(
                    (pos + np.array(n) @ self.supercell_vectors, s, idx))

    def position_key(self, pos: np.ndarray) -> tuple[float, ...]:
        '''
        Fractional coordinates of *pos* in the supercell basis, modulo one cell.

        Args:
            pos: Cartesian position.
        Returns:
            The fractional coordinates, each reduced into ``[0, 1)`` and rounded
            to ``_POS_TOL_DIGITS`` decimals.  Positions that differ by a
            supercell-lattice vector give the same key.
        '''
        frac = np.asarray(pos, dtype=float) @ np.linalg.inv(self.supercell_vectors)
        frac = frac - np.floor(frac + 1e-9)
        return tuple(np.round(frac, _POS_TOL_DIGITS))

    def site_index(self, pos: np.ndarray, s: str) -> int:
        '''
        Index of the site of type *s* sitting at *pos*, modulo the supercell lattice.

        Args:
            pos: Cartesian position of a site, up to a supercell-lattice vector.
            s: site type.
        Raises:
            ValueError: if no site of type *s* sits at that position.
        '''
        key = (s, self.position_key(pos))
        if key not in self._site_lookup:
            raise ValueError(f"No site of type {s!r} at {np.asarray(pos).round(6).tolist()}.")
        return self._site_lookup[key]

    def plot(self, ax: Optional[plt.Axes] = None, extend: bool = False) -> plt.Axes:
        '''
        Plot the lattice.

        Args:
            ax: the axes to draw on; a new figure is made when None.
            extend: also draw the periodically extended copy.
        Returns:
            The axes.
        '''
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        if extend:
            for pos, _, idx in self.extended_cell:
                ax.scatter(*pos, color='blue', alpha=0.6)
                ax.text(*pos, str(idx), fontsize=8, ha='center', va='center')
        for pos, _, idx in self.unit_cell_idx:
            ax.scatter(*pos, color='red', alpha=0.8)
            ax.text(*pos, str(idx), fontsize=10, ha='center', va='center', color='white')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Lattice Structure')
        return ax

    def set_center_at(self, center: int | np.ndarray):
        '''
        Shift the origin to a site index or to a position.

        Args:
            center: index of a site, or a Cartesian position to put at the origin.
        Raises:
            ValueError: if the index is out of range.
        '''
        if isinstance(center, int):
            if not (0 <= center < self.size_of_unit_cell):
                raise ValueError(f"Center index must be between 0 and {self.size_of_unit_cell-1}.")
            center_pos, _, _ = self.unit_cell_idx[center]
        else:
            center_pos = np.asarray(center)
        self.extended_cell = [(pos - center_pos, s, idx) for pos, s, idx in self.extended_cell]
        self.unit_cell_idx = [(pos - center_pos, s, idx) for pos, s, idx in self.unit_cell_idx]


@dataclass(frozen=True)
class AffineAction(LatticeAction):
    '''
    One element of the space group:  x -> R x + t.

    ``translation`` is the fractional part of the operation.  It vanishes for
    every element of a symmorphic space group; it is what makes a glide plane
    or a screw axis expressible, since such an operation is a rotation plus a
    non-lattice translation.

    Attributes:
        rotation: ``(d, d)`` orthogonal matrix.
        translation: ``(d,)`` translation, reduced modulo the supercell lattice.
    '''
    rotation: np.ndarray
    translation: np.ndarray

    @staticmethod
    def from_matrix(op: np.ndarray, spatial_dim: int) -> 'AffineAction':
        '''
        Build an action from the bare coordinate matrix stored by a point group.

        Args:
            op: a ``(d, d)`` matrix, or a scalar for the trivial group.
            spatial_dim: expected spatial dimension.
        Returns:
            The action, with a zero translation.
        Raises:
            ValueError: if *op* does not have the expected shape.
        '''
        arr = np.asarray(op, dtype=float)
        if arr.ndim == 0:
            return AffineAction(np.eye(spatial_dim), np.zeros(spatial_dim))
        if arr.shape != (spatial_dim, spatial_dim):
            raise ValueError(
                f"Transformation matrix must be of shape ({spatial_dim}, {spatial_dim}), "
                f"got {arr.shape}.")
        return AffineAction(arr, np.zeros(spatial_dim))

    def __matmul__(self, other: 'AffineAction') -> 'AffineAction':
        '''``(R1,t1)(R2,t2) = (R1 R2, t1 + R1 t2)``.'''
        if not isinstance(other, AffineAction):
            raise TypeError("Can only compose with another AffineAction.")
        return AffineAction(self.rotation @ other.rotation,
                            self.translation + self.rotation @ other.translation)

    def inverse(self) -> 'AffineAction':
        '''The inverse element ``(R,t)^{-1} = (Rᵀ, -Rᵀt)``.'''
        return AffineAction(self.rotation.T, -self.rotation.T @ self.translation)

    def apply(self, pos: np.ndarray) -> np.ndarray:
        '''``x -> R x + t``.'''
        return self.rotation @ np.asarray(pos, dtype=float) + self.translation

    def site_map(self, lattice: Lattice) -> np.ndarray:
        '''
        Permutation of the sites of *lattice*.

        Args:
            lattice: the lattice the operation acts on.
        Returns:
            ``image[i]``: the site that site ``i`` is mapped onto.
        Raises:
            ValueError: if the operation does not map the sites onto the sites,
                or if it maps two sites onto the same one -- either way it is
                not a symmetry of the finite lattice.
        '''
        image = np.zeros(lattice.size_of_unit_cell, dtype=int)
        for pos, s, idx in lattice.unit_cell_idx:
            try:
                image[idx] = lattice.site_index(self.apply(pos), s)
            except ValueError:
                raise ValueError(
                    f"Operation {self.rotation.tolist()} does not map the site {idx} at "
                    f"{np.asarray(pos).round(6).tolist()} onto a site of the supercell: it is "
                    f"not a symmetry of this lattice. Either enlarge the supercell or drop "
                    f"this operation.") from None
        if not np.array_equal(np.sort(image), np.arange(lattice.size_of_unit_cell)):
            raise ValueError(
                f"Operation {self.rotation.tolist()} maps several sites onto the same site: it "
                f"does not preserve the supercell lattice spanned by "
                f"{lattice.supercell_vectors.tolist()}.")
        return image

    def label_action(self, supercell_vectors: np.ndarray) -> np.ndarray:
        '''Integer action on the momentum labels, see :func:`label_action`.'''
        return label_action(self.rotation, supercell_vectors)

    def preserves_momentum(self, k: tuple[int, ...], supercell_vectors: np.ndarray,
                           Nvec: np.ndarray) -> bool:
        '''Whether this operation leaves the momentum label *k* invariant.'''
        return preserves_momentum(self.rotation, k, supercell_vectors, Nvec)
