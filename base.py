import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from itertools import product

class Lattice:
    '''
    Class representing a lattice structure. 
    '''
    def __init__(self, 
                 supercell_info:list[tuple[np.ndarray,int]],
                 cell_info: dict[tuple[str,int], np.ndarray], 
                 point_group: PointGroup|None = None,
                 MAX_EXTENSION = 1
                 ):
        '''
        Initialize a Lattice object. A lattice (supercell) is defined by its reciprocal vectors and their multiplicities, as well as additional cell information.
        
        Args:
            supercell_info: A variable number of tuples, each containing a supercell vector (as a numpy array) and its multiplicity (as an integer). Eventually, the supercell vectors will be multiplied by their respective multiplicities to form the complete supercell.
            cell_info: Additional information about the cell, provided as keyword arguments. Each key is a tuple of (atom_type, atom_index), and each value is a numpy array representing the atom's position in the cell.
            MAX_EXTENSION: The maximum extension for the periodic boundary conditions.
        '''
        self.spacial_dim = len(supercell_info[0])

        if not all(isinstance(vec, np.ndarray) for vec, _ in supercell_info):
            raise TypeError("All supercell vectors must be numpy arrays.")
        if not all(len(vec) == self.spacial_dim for vec, _ in supercell_info):
            raise ValueError("All supercell vectors must have the same dimension {}.".format(self.spacial_dim))
        if not all(isinstance(m, int) and m > 0 for _, m in supercell_info):
            raise ValueError("All multiplicities must be positive integers.")

        self.supercell_vectors =  np.array([vec*m for vec, m in supercell_info])
        self.cell_vectors = np.array([vec for vec, _ in supercell_info])

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

        

        
        

    def set_center_at(self, center:int | np.ndarray):
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

    def plot(self, ax=None):
        '''
        Plot the lattice structure.
        Args:
            ax: A matplotlib Axes object. If None, a new figure and axes will be created.
        '''
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        
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

    def _get_permutation_pt(self, arr:np.ndarray)-> Permutation:
        '''
        Given a point group transformation arr, return the permutation of the atoms in the unit cell that corresponds to this transformation.
        '''
        assert arr.shape == (self.spacial_dim, self.spacial_dim), \
        f"Transformation matrix must be of shape ({self.spacial_dim}, {self.spacial_dim})."

        permuted = np.zeros(self.size_of_unit_cell, dtype=int)

        for pos, s, idx in self.unit_cell_idx:
            transformed_pos = arr @ pos
            permuted[idx] = self._get_idx(transformed_pos, s)
        return Permutation(permuted)
    def _get_permutation_tr(self, dir:int)-> Permutation:
        '''
        Given a translation vector dir, return the permutation of the atoms in the unit cell that corresponds to this translation.
        '''
        assert 0 <= dir < self.spacial_dim, f"Translation direction must be between 0 and {self.spacial_dim-1}."

        translated = np.zeros(self.size_of_unit_cell, dtype=int)
        for pos, s, idx in self.unit_cell_idx:
            translated[idx] = self._get_idx(pos + self.cell_vectors[dir], s)
        return Permutation(translated)

    
class Permutation:
    '''
    Class representing a permutation of atoms in the lattice.
    '''
    def __init__(self, perm: np.ndarray):
        self.perm = perm
        self.size = len(perm)
        if not np.array_equal(np.sort(perm), np.arange(self.size)):
            raise ValueError("Invalid permutation array. It must contain all integers from 0 to size-1 exactly once.")

    def __call__(self, x:np.ndarray) -> np.ndarray:
        return x[self.perm]

    def __mul__(self, other: 'Permutation') -> 'Permutation':
        if not isinstance(other, Permutation):
            raise TypeError("Can only multiply with another Permutation.")
        if self.size != other.size:
            raise ValueError("Permutations must be of the same size to multiply.")
        return Permutation(self.perm[other.perm])

    def __hash__(self):
        return hash(tuple(self.perm))




class PointGroup(ABC):
    '''
    Class representing a group of rotations and inversion. 
    '''
    @abstractmethod
    def little_group(self, k:np.ndarray) -> tuple['PointGroup', list[np.ndarray]]:
        '''
        Return the little group of a given momentum k.
        Args:
            k: A numpy array representing the momentum vector.
        Returns:
            (grp,lst): A PointGroup object representing the little group of k, and a list representing the left coset representatives of the little group in the full point group.
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
    def __iter__(self):
        '''
        Return an iterator over the group operations in the point group.
        '''
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def __next__(self):
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
        ...

    def all_irrps(self) -> list[str]:
        return ['A']
    def little_group(self, k:np.ndarray) -> tuple['PointGroup', list[np.ndarray]]:
        return self, [np.eye(len(k))]

    def character(self, irrp:str, op:np.ndarray) -> complex:
        if irrp == 'A':
            return 1.0
        else:
            raise ValueError(f"Unknown irreducible representation: {irrp}")

    def __iter__(self):
        return self

    def __next__(self):
        yield 1
        raise StopIteration