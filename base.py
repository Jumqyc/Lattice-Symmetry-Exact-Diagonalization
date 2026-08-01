from abc import ABC, abstractmethod
from typing import Optional
from functools import reduce
from itertools import product

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

class Model:
    '''
    Class representing a lattice structure. 
    '''
    def __init__(self, 
                 phys_dim:int,
                 supercell_info:list[tuple[np.ndarray,int]],
                 cell_info: dict[tuple[str,int], np.ndarray], 
                 point_group: Optional[PointGroup] = None,
                 MAX_EXTENSION = 1
                 ):
        '''
        Initialize a Lattice object. A lattice (supercell) is defined by its reciprocal vectors and their multiplicities, as well as additional cell information.
        
        Args:
            phys_dim: The physical dimension of the lattice.
            supercell_info: A variable number of tuples, each containing a supercell vector (as a numpy array) and its multiplicity (as an integer). Eventually, the supercell vectors will be multiplied by their respective multiplicities to form the complete supercell.
            cell_info: Additional information about the cell, provided as keyword arguments. Each key is a tuple of (atom_type, atom_index), and each value is a numpy array representing the atom's position in the cell.
            point_group: An optional PointGroup object representing the symmetries of the lattice. If not provided, a trivial point group will be used.
            MAX_EXTENSION: The maximum extension for the periodic boundary conditions.
        '''
        self.spacial_dim = len(supercell_info[0])
        self.physical_dim = phys_dim

        if not all(isinstance(vec, np.ndarray) for vec, _ in supercell_info):
            raise TypeError("All supercell vectors must be numpy arrays.")
        if not all(len(vec) == self.spacial_dim for vec, _ in supercell_info):
            raise ValueError("All supercell vectors must have the same dimension {}.".format(self.spacial_dim))
        if not all(isinstance(m, int) and m > 0 for _, m in supercell_info):
            raise ValueError("All multiplicities must be positive integers.")

        self.cell_vectors: NDArray[np.float64] = np.array([vec for vec, _ in supercell_info])
        self.Nvec: NDArray[np.int64] = np.array([m for _, m in supercell_info])
        self.supercell_vectors: NDArray[np.float64] = self.cell_vectors * self.Nvec[:, np.newaxis]

        if not self.cell_vectors.shape == (self.spacial_dim, self.spacial_dim):
            raise ValueError("The number of supercell vectors must match the spatial dimension.")

        self.bz_cellvectors = 2*np.pi*np.linalg.inv(self.supercell_vectors).T

        self.unit_cell_idx: list[tuple[np.ndarray,str, int]] = []
        for n in product(*[range(m) for _, m in supercell_info]):
            for (s,_),vec in cell_info.items():
                self.unit_cell_idx.append(
                    (vec + np.array(n) @ self.cell_vectors ,
                    s,
                    len(self.unit_cell_idx))) 
                    # centering the cell at the origin and assigning a unique index to each atom
        self.size_of_unit_cell = len(self.unit_cell_idx)

        # extending by PBC:
        self.extended_cell = []
        # dumb implementation by storing all possible positions
        for pos, s, idx in self.unit_cell_idx: 
            # to avoid modifying the list while iterating over it
            for n in product(
                        range(-MAX_EXTENSION, MAX_EXTENSION + 1),
                        repeat=self.spacial_dim):
                self.extended_cell.append(
                    (pos + np.array(n) @ self.supercell_vectors, s, idx)
                )

        if point_group is None:
            self.point_group = TrivialGroup()
        else:
            self.point_group = point_group

        self.all_sym: dict[tuple[int,...], SymmetryOperation] = {}

        for n in product(*[range(Ni) for Ni in self.Nvec]):
            for i,r in enumerate(self.point_group):
                perm = self._get_permutation_pt(r) * reduce(
                    lambda x, y: x * y, 
                    [self._get_permutation_tr(ni) for ni in n])
                self.all_sym[tuple(n) + (i,) + (1,)] = perm

                perm_f = perm.copy()
                perm_f.spin_flip = True
                self.all_sym[tuple(n) + (i,) + (-1,)] = perm_f


    def add_coupling(self, *coupling:dict[int, np.ndarray]):
        r'''
        Add coupling terms to the model. Each coupling term is represented as a dictionary mapping atom indices to their corresponding coupling matrices.

        For the coupling, it is assumed that the coupling will respect symmetry. In other words, the true Hamiltonian should be H_tot = 1/N \sum_{g in G} g H_0 g^{-1}, where N is the number of stabilizer elements of the coupling term. 
        
        Args:
            *coupling: A variable number of dictionaries, each representing a coupling term. The keys are the atom indices (as integers), and the values are numpy arrays representing the coupling matrices. It is assumed that the coupling matrices are of shape (phys_dim, phys_dim). The operators will be multiplied in each dictionary in the order of the keys, and then summed over all dictionaries.
        Example:
            >>> X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.array([[1, 0], [0, -1]])
            >>> model.add_coupling({0: X, 1: Y}, {0: Z})
            # this will add X_0 * Y_1 + Z_0 to the model.
        '''
        pass
        



        
    def momentum_vec(self, *n:int):
        '''
        Compute the momentum vector in the Brillouin zone given a set of indices.
        Args:
            *n: A variable number of indices corresponding to the supercell vectors.
        Returns:
            A numpy array representing the momentum vector in the Brillouin zone.
        '''
        if len(n) != self.spacial_dim:
            raise ValueError(f"Expected {self.spacial_dim} indices, got {len(n)}.")
        if not all((0 <= ni < Ni) for ni, Ni in zip(n, self.Nvec)):
            raise ValueError(f"Indices must in the Brillouin zone. The multiplicities are {self.Nvec}, but got indices {n}.")
        return np.array(n) @ self.bz_cellvectors
        
    # TODO: rewrite this in C++ and bind it to Python.
    def _build_adapt_basis(self,
                          N:Optional[int] = None
                          )-> dict[tuple[int],int]:
        '''
        Build an adapted basis for the lattice, taking into account the symmetries of the system
        Args:
            N: An optional integer representing the total number of particles. If provided, only states with this total number of particles will be included in the adapted basis.
        Returns:
            A dictionary mapping each unique state (as a tuple of integers) to its corresponding index in the adapted basis.
        '''

        basis = {}
        if N is None: # if N is not provided, include all states
            for state in product(range(self.physical_dim), repeat=self.size_of_unit_cell):
                for s in self.all_sym.values():
                    permuted_state = tuple(s(np.array(state)))
                    if permuted_state not in basis:
                        basis[permuted_state] = len(basis)
        else: # if N is provided, only include states with total number of particles equal to N
            for state in product(range(self.physical_dim), repeat=self.size_of_unit_cell):
                if sum(state) != N:
                    continue
                for s in self.all_sym.values():
                    permuted_state = tuple(s(np.array(state)))
                    if permuted_state not in basis:
                        basis[permuted_state] = len(basis)

        return basis


    def set_center_at(self, center: int | np.ndarray):
        '''
        Center the lattice at a specific atom index or position.
        Args:
            center: The index of the atom (int), or a numpy array position,
                    to center the lattice around.
        '''
        if isinstance(center, int):
            if not (0 <= center < self.size_of_unit_cell):
                raise ValueError(f"Center index must be between 0 and {self.size_of_unit_cell-1}.")
            center_pos, _, _ = self.unit_cell_idx[center]
        else:
            center_pos = np.asarray(center)

        self.extended_cell = [(pos - center_pos, s, idx) for pos, s, idx in self.extended_cell]
        self.unit_cell_idx = [(pos - center_pos, s, idx) for pos, s, idx in self.unit_cell_idx]

    def plot(self, 
             ax:Optional[plt.Axes]=None,
             extend:bool = False) -> plt.Axes:
        '''
        Plot the lattice structure.
        Args:
            ax: A matplotlib Axes object. If None, a new figure and axes will be created.
            extend: If True, plot the extended cell; otherwise, plot only the unit cell.
        '''
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        if extend:
            for pos, _, idx in self.extended_cell:
                ax.scatter(*pos, alpha=0.6,color='blue')
                ax.text(*pos, str(idx), fontsize=8, ha='center', va='center')

        for pos, _, idx in self.unit_cell_idx:
            ax.scatter(*pos, color='red', alpha=0.8)
            ax.text(*pos, str(idx), fontsize=10, ha='center', va='center', color='white')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Lattice Structure')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        return ax

    def _get_idx(self, pos:np.ndarray,s:str,tol = 1e-6) -> int:
        # brute force search
        # self.extended_cell is at most 1000, so this is fine
        for p,s1,idx in self.extended_cell:
            if np.allclose(p, pos, atol=tol) and s1 == s:
                return idx
        raise ValueError(f"Position {pos} not found in unit cell nor extended cell.")

    def _get_permutation_pt(self, arr:np.ndarray)-> SymmetryOperation:
        '''
        Given a point group transformation arr, return the permutation of the atoms in the unit cell that corresponds to this transformation.
        Accepts either a (d,d) matrix or a 0-d scalar (identity for any dimension).
        '''
        if arr.ndim == 0:
            # TrivialGroup identity: scalar 1 → acts as identity for any dimension
            pass
        else:
            assert arr.shape == (self.spacial_dim, self.spacial_dim), \
            f"Transformation matrix must be of shape ({self.spacial_dim}, {self.spacial_dim}), got {arr.shape}."

        permuted = np.zeros(self.size_of_unit_cell, dtype=int)

        for pos, s, idx in self.unit_cell_idx:
            if arr.ndim == 0:
                transformed_pos = arr * pos       # scalar identity
            else:
                transformed_pos = arr @ pos
            permuted[idx] = self._get_idx(transformed_pos, s)
        return SymmetryOperation(permuted, 
                                 physical_dim=self.physical_dim, 
                                 spin_flip=False)
    
    def _get_permutation_tr(self, dir:int)-> SymmetryOperation:
        '''
        Given a translation vector dir, return the permutation of the atoms in the unit cell that corresponds to this translation.
        '''
        assert 0 <= dir < self.spacial_dim, f"Translation direction must be between 0 and {self.spacial_dim-1}."

        translated = np.zeros(self.size_of_unit_cell, dtype=int)
        for pos, s, idx in self.unit_cell_idx:
            translated[idx] = self._get_idx(pos + self.cell_vectors[dir], s)
        return SymmetryOperation(translated, physical_dim=self.physical_dim, spin_flip=False)

    
class SymmetryOperation:
    '''
    Class representing a symmetry operation on atoms in the lattice.
    '''
    def __init__(self, 
                 perm: np.ndarray,
                 physical_dim: int,
                 spin_flip: bool = False):
        self.perm = perm
        self.size = len(perm)
        self.spin_flip = spin_flip
        self.physical_dim = physical_dim
        if not np.array_equal(np.sort(perm), np.arange(self.size)):
            raise ValueError("Invalid permutation array. It must contain all integers from 0 to size-1 exactly once.")

    def __call__(self, x:np.ndarray) -> np.ndarray:
        if not self.spin_flip:
            return x[self.perm]
        else:
            return self.physical_dim - 1 - x[self.perm] 

    def __mul__(self, other: 'SymmetryOperation') -> 'SymmetryOperation':
        if not isinstance(other, SymmetryOperation):
            raise TypeError("Can only multiply with another SymmetryOperation.")
        if self.size != other.size:
            raise ValueError("SymmetryOperations must be of the same size to multiply.")
        return SymmetryOperation(self.perm[other.perm], 
                                 physical_dim=self.physical_dim, 
                                 spin_flip=self.spin_flip ^ other.spin_flip)
    def inv(self)->SymmetryOperation:
        inv_perm = np.argsort(self.perm)
        return SymmetryOperation(inv_perm,
                                 physical_dim=self.physical_dim,
                                 spin_flip=self.spin_flip)
    def __pow__(self, power: int) -> 'SymmetryOperation':
        if power == 0:
            return SymmetryOperation(np.arange(self.size), self.physical_dim, self.spin_flip)
        elif power > 0:
            result = self
            for _ in range(power - 1):
                result = result * self
            return result
        else:
            return self.inv() ** (-power)
    def copy(self) -> SymmetryOperation:
        return SymmetryOperation(self.perm.copy(), self.physical_dim, self.spin_flip)

    def __hash__(self):
        return hash(tuple(self.perm))


class PointGroup(ABC):
    '''
    Class representing a group of rotations and inversion. 
    '''
    @abstractmethod
    def little_group(self, k:np.ndarray,
                     reciprocal_basis:Optional[np.ndarray] = None
                     ) -> tuple['PointGroup', list[np.ndarray]]:
        '''
        Return the little group of a given momentum k.

        Args:
            k: A numpy array representing the momentum vector.
            reciprocal_basis: Optional (d,d) array whose columns are the
                reciprocal-lattice basis vectors of the supercell.  When
                provided, *op* ∈ G_pt belongs to the little group iff
                ``op @ k ≈ k + reciprocal_basis @ n`` for some integer
                vector *n*.  When ``None``, exact preservation
                ``op @ k ≈ k`` is used (suitable for Γ, or when the
                reciprocal lattice is unknown).

        Returns:
            (grp,lst): A PointGroup object representing the little group
            of k, and a list of left-coset representatives
            G_pt / G_k.
        '''
        raise NotImplementedError("This method should be implemented in subclasses.")
        
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
                  irrp:str, 
                  op:np.ndarray) -> complex:
        '''
        Return the character of a given irreducible representation for a specific group operation.
        Args:
            irrp: A string representing the label of the irreducible representation.
            op: A numpy array representing the group operation (rotation or inversion).
        Returns:
            A complex representing the character of the irreducible representation for the given operation.
        '''
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def __iter__(self)-> PointGroup:
        '''
        Return an iterator over the group operations in the point group.
        '''
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def __next__(self) -> np.ndarray:
        '''
        Return the next group operation in the point group.
        '''
        raise NotImplementedError("This method should be implemented in subclasses.")


class TrivialGroup(PointGroup):
    '''
    A trivial point group containing only the identity operation.

    The only irrp of this group is 'A', which has a character of 1 for the identity operation.
    '''
    def __init__(self):
        self._iterated = False

    def all_irrps(self) -> list[str]:
        return ['A']

    def little_group(self, k:np.ndarray,
                     reciprocal_basis:Optional[np.ndarray] = None
                     ) -> tuple['PointGroup', list[np.ndarray]]:
        return self, [np.eye(len(k))]

    def character(self, irrp:str, op:np.ndarray) -> complex:
        if irrp == 'A':
            return 1.0
        else:
            raise ValueError(f"Unknown irreducible representation: {irrp}")

    def __iter__(self):
        self._iterated = False
        return self

    def __next__(self) -> np.ndarray:
        if self._iterated:
            raise StopIteration
        self._iterated = True
        return np.array(1)