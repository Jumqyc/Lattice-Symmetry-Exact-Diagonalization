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

    Three implementations answer it, and they answer it for different reasons.
    :class:`representation.TrivialGroup` is the default, the group that says the
    model has no point symmetry.  :class:`representation.ExplicitPointGroup` reads
    the table off an operation list, which is the only thing possible for the little
    groups: they are subgroups of the point group at one momentum, so they are not
    any of the named groups.  The named ones in ``geometry/2dsym.py`` override the
    table to keep their own labels and conventions.  A model holds one of these
    three and never learns which.
    '''

    @abstractmethod
    def all_irrps(self) -> list[str]:
        '''
        Return a list of all irreducible representations of the point group.
        Returns:
            A list of strings representing the labels of all irreducible representations.
        '''
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    @abstractmethod
    def character(self, 
                  irrep:str, 
                  op:np.ndarray) -> complex:
        '''
        Return the character of a given irreducible representation for a specific group operation.
        Args:
            irrep: A string representing the label of the irreducible representation.
            op: A numpy array representing the group operation (rotation or inversion).
        Returns:
            A complex representing the character of the irreducible representation for the given operation.
        '''
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        '''
        Return a fresh iterator over the group operations.

        Iteration must not carry state: the same group object is iterated
        several times, sometimes nested.
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
