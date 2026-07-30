import numpy as np
import matplotlib.pyplot as plt

from itertools import product

class Lattice:
    '''
    Class representing a lattice structure. 
    '''
    def __init__(self, 
                 supercell_info:list[tuple[np.ndarray,int]],
                 cell_info: dict[tuple[str,int], np.ndarray], 
                 MAX_EXTENSION = 1
                 ):
        '''
        Initialize a Lattice object. A lattice (supercell) is defined by its reciprocal vectors and their multiplicities, as well as additional cell information.
        Arguments:
        - supercell_info: A variable number of tuples, each containing a supercell vector (as a numpy array) and its multiplicity (as an integer). Eventually, the supercell vectors will be multiplied by their respective multiplicities to form the complete supercell.
        - cell_info: Additional information about the cell, provided as keyword arguments. Each key is a tuple of (atom_type, atom_index), and each value is a numpy array representing the atom's position in the cell.
        - tol: Tolerance for numerical comparisons. 
        '''
        self.spacial_dim = len(supercell_info[0])

        assert all(len(vec) == self.spacial_dim for vec, _ in supercell_info), "All supercell vectors must have the same dimension {}.".format(self.spacial_dim)
        assert all(isinstance(m, int) and m > 0 for _, m in supercell_info), "All multiplicities must be positive integers."

        self.supercell_vectors =  np.array([vec*m for vec, m in supercell_info])
        self.reciprocal_vectors = np.array([vec for vec, _ in supercell_info])

        self.unit_cell_idx: list[tuple[np.ndarray,str, int]] = []
        for n in product(*[range(m) for _, m in supercell_info]):
            for (s,_),vec in cell_info.items():
                self.unit_cell_idx.append(
                    (vec + np.array(n) @ self.reciprocal_vectors ,
                    s,
                    len(self.unit_cell_idx))) 
                    # centering the cell at the origin and assigning a unique index to each atom
        self.size_of_unit_cell = len(self.unit_cell_idx)


        # extending by PBC:
        self.extended_cell = []
        for pos, s, idx in self.unit_cell_idx: # to avoid modifying the list while iterating over it
            for n in product(
                        range(-MAX_EXTENSION, MAX_EXTENSION + 1),
                        repeat=self.spacial_dim):
                self.extended_cell.append(
                    (pos + np.array(n) @ self.supercell_vectors, s, idx)
                )

    def set_center_at(self, center):
        '''
        Center the lattice at a specific atom index or position.
        Arguments:
        - center: The index of the atom (int), or a numpy array position,
                  to center the lattice around.
        '''
        if isinstance(center, int):
            assert 0 <= center < self.size_of_unit_cell, f"Center index must be between 0 and {self.size_of_unit_cell-1}."
            center_pos, _, _ = self.unit_cell_idx[center]
        else:
            center_pos = np.asarray(center)
        self.extended_cell = [(pos - center_pos, s, idx) for pos, s, idx in self.extended_cell]
        self.unit_cell_idx = [(pos - center_pos, s, idx) for pos, s, idx in self.unit_cell_idx]

    def plot(self, ax=None):
        '''
        Plot the lattice structure.
        Arguments:
        - ax: A matplotlib Axes object. If None, a new figure and axes will be created.
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        for pos, s, idx in self.extended_cell:
            ax.scatter(*pos, alpha=0.6)
            ax.text(*pos, str(idx), fontsize=8, ha='center', va='center')

        for pos, s, idx in self.unit_cell_idx:
            ax.scatter(*pos, color='red', alpha=0.8)
            ax.text(*pos, str(idx), fontsize=10, ha='center', va='center', color='white')
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Lattice Structure')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        return ax


    def _get_idx(self, pos,s,tol = 1e-6) -> int:
        for p,s1,idx in self.extended_cell:
            if np.allclose(p, pos, atol=tol) and s1 == s:
                return idx
        raise ValueError(f"Position {pos} not found in unit cell nor extended cell.")

    def get_permutation_pt(self, arr:np.ndarray)-> np.ndarray:
        '''
        Given a point group transformation arr, return the permutation of the atoms in the unit cell that corresponds to this transformation.
        '''
        assert arr.shape == (self.spacial_dim, self.spacial_dim), \
        f"Transformation matrix must be of shape ({self.spacial_dim}, {self.spacial_dim})."

        permuted = np.zeros(self.size_of_unit_cell, dtype=int)

        for pos, s, idx in self.unit_cell_idx:
            transformed_pos = arr @ pos
            permuted[idx] = self._get_idx(transformed_pos, s)
        if np.array_equal(np.sort(permuted), np.arange(self.size_of_unit_cell)) == False:
            raise ValueError(f"{arr} is not a valid permutation, as permuted = {np.sort(permuted)}.")
        
        return permuted
    def get_permutation_tr(self, dir):
        '''
        Given a translation vector dir, return the permutation of the atoms in the unit cell that corresponds to this translation.
        '''
        assert 0 <= dir < self.spacial_dim, f"Translation direction must be between 0 and {self.spacial_dim-1}."

        translated = np.zeros(self.size_of_unit_cell, dtype=int)
        for pos, s, idx in self.unit_cell_idx:
            translated[idx] = self._get_idx(pos + self.reciprocal_vectors[dir], s)
        return translated

    








        