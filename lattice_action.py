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

from itertools import product
from typing import Iterator

import numpy as np

from base import PointGroup
from representation import (ExplicitPointGroup, TrivialGroup, _right_coset_representatives,
                            _unique_ops)


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
