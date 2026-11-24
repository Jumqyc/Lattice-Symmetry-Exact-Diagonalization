'''
Interfaces of the symmetry machinery.

The symmetry part of the framework is split in three layers, and this module
holds nothing but the interface of the first one:

* :class:`PointGroup` -- representation theory: the elements of the group, the
  multiplication (through the element matrices) and the character table.  It
  knows nothing about the lattice and nothing about the Hilbert space.
* ``LatticeAction`` -- the geometry: how one element moves the sites and the
  momentum labels (``lattice_action.py``).
* ``HilbertAction`` -- the action on the computational basis of the Hilbert
  space (``hilbert_action.py``).

Keeping the interface here and the implementations in separate modules keeps
the dependency one-directional: ``base`` -> ``representation`` ->
``lattice_action`` -> ``hilbert_action`` -> ``ed``.
'''

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class PointGroup(ABC):
    '''
    A finite group, described only through its elements and their characters.

    The interface is three methods -- the operations, the irrep labels, and the
    character of one irrep at one operation -- because that is all the
    decomposition asks: it iterates the group, sums characters, and never needs to
    know which group it has.  The quantities that follow from the interface --
    the identity, the dimension of an irrep -- are computed here, once.

    The implementations answer it in two ways.  :class:`representation.TrivialGroup`
    is the default, the group that says the model has no point symmetry, and
    :class:`representation.CyclicGroup` / :class:`representation.DihedralGroup` are
    the two kinds a finite subgroup of the plane can be, each with its own table;
    :func:`representation.group_of` builds the kind a list of operations turns out to
    be, which is what the little groups are, since they are subgroups of the point
    group at one momentum rather than any of the named groups.  The named ones in
    ``geometry/2dsym.py`` inherit those two and override the table to keep their own
    labels and conventions.  A model holds one of these and never learns which.
    '''

    # a class may rename the labels its table is written for; empty means it does not
    _labels: dict[str, str] = {}

    def all_irrps(self) -> list[str]:
        '''
        Return the labels of the irreducible representations, in this group's naming.

        Returns:
            The labels, in the order the table is written.
        '''
        return list(self._labels) if self._labels else self._irrep_labels()

    def character(self, irrep: str, op: np.ndarray) -> complex:
        '''
        Return the character of one irreducible representation at one operation.

        Args:
            irrep: one of the labels ``all_irrps`` returns.
            op: a group operation, as a matrix.
        Returns:
            The character.
        Raises:
            ValueError: if *irrep* is not a label of this group.
        '''
        label = self._labels[irrep] if self._labels and irrep in self._labels else irrep
        if label not in self._irrep_labels():
            raise ValueError(f"Unknown irreducible representation: {irrep}")
        return self._character(label, op)

    @abstractmethod
    def _irrep_labels(self) -> list[str]:
        '''
        The labels the character table of this group is written for.
        '''
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def _character(self, irrep: str, op: np.ndarray) -> complex:
        '''
        The character of one of the table's own labels at one operation.

        Args:
            irrep: a label ``_irrep_labels`` returns.
            op: a group operation, as a matrix.
        Returns:
            The character.
        '''
        raise NotImplementedError("This method should be implemented in subclasses.")

    # ── derived quantities ───────────────────────────────────────────────

    def identity(self) -> np.ndarray:
        '''The identity operation of the group.'''
        for op in self:
            arr = np.asarray(op, dtype=float)
            if arr.ndim == 0 or np.allclose(arr, np.eye(arr.shape[0])):
                return arr
        raise ValueError("The group does not contain the identity operation.")

    def irrep_dim(self, irrep: str) -> int:
        '''
        Dimension of an irrep, i.e. its character on the identity.

        Args:
            irrep: label of the irrep.
        Returns:
            The dimension as an integer.
        '''
        return int(round(float(np.real(self.character(irrep, self.identity())))))


class LatticeAction(ABC):
    '''
    One element of the space group, seen as a map on the lattice.

    This is the geometry layer: an element knows how to move the sites and how
    to move the momentum labels.  It knows nothing about the Hilbert space --
    turning it into an operator on the computational basis is the job of
    :class:`HilbertAction`.
    '''

    @abstractmethod
    def __matmul__(self, other: 'LatticeAction') -> 'LatticeAction':
        '''Group multiplication: apply *other* first, then *self*.'''

    @abstractmethod
    def site_map(self, lattice) -> np.ndarray:
        '''
        Permutation of the sites of *lattice*.

        Returns:
            ``image[i]`` is the site that site ``i`` is mapped onto.
        Raises:
            ValueError: if the operation does not map the sites onto the sites.
        '''

    @abstractmethod
    def label_action(self, supercell_vectors: np.ndarray) -> np.ndarray:
        '''
        Integer matrix M acting on the momentum labels, ``k -> Mᵀ k``.
        '''


class HilbertAction(ABC):
    '''
    One symmetry operation, seen as a map on the computational basis.

    The interface is deliberately narrow: an operation maps one basis state
    onto *one* basis state, times a phase,

        U(g)|σ> = φ(σ) |σ'>.

    That is what keeps every projection in the computational basis cheap.  An
    operation whose local action mixes components is not expressible here, and
    it must not be: it would turn a single basis state into an exponential
    superposition.  A monomial local basis -- in two dimensions the circular
    harmonics ``(x ± iy)^m`` -- always avoids it.
    '''

    @abstractmethod
    def on_state(self, state: tuple[int, ...]) -> tuple[tuple[int, ...], complex]:
        '''``U(g)|state> = phase |state'>``; returns ``(state', phase)``.'''
