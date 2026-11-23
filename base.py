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

    Subclasses provide the operations, their irreps and their characters.  The
    geometry of the lattice and the decomposition of the momenta are handled by
    the lattice layer, not here.
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

    def order(self) -> int:
        '''Number of operations of the group.'''
        return len(list(self))

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
