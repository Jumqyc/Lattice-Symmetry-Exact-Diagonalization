from abc import ABC, abstractmethod
from functools import reduce
from itertools import product
from typing import Optional, Iterator

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray



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
        self.spatial_dim = len(supercell_info[0])
        self.physical_dim = phys_dim

        if not all(isinstance(vec, np.ndarray) for vec, _ in supercell_info):
            raise TypeError("All supercell vectors must be numpy arrays.")
        if not all(len(vec) == self.spatial_dim for vec, _ in supercell_info):
            raise ValueError("All supercell vectors must have the same dimension {}.".format(self.spatial_dim))
        if not all(isinstance(m, int) and m > 0 for _, m in supercell_info):
            raise ValueError("All multiplicities must be positive integers.")

        self.cell_vectors: NDArray[np.float64] = np.array([vec for vec, _ in supercell_info])
        self.Nvec: NDArray[np.int64] = np.array([m for _, m in supercell_info])
        self.supercell_vectors: NDArray[np.float64] = self.cell_vectors * self.Nvec[:, np.newaxis]

        if not self.cell_vectors.shape == (self.spatial_dim, self.spatial_dim):
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
                        repeat=self.spatial_dim):
                self.extended_cell.append(
                    (pos + np.array(n) @ self.supercell_vectors, s, idx)
                )

        if point_group is None:
            self.point_group = TrivialGroup()
        else:
            self.point_group = point_group


        self.all_perms = [self._get_permutation_pt(op) * 
                          reduce(lambda x, y: x * y, 
                            [self._get_permutation_tr(i) ** ni for i, ni in enumerate(n)])
                          for op in self.point_group 
                          for n in product(*self.Nvec)]

        self.couplings: list[Coupling] = []
    
    def add_coupling(self,
                     coupling: Coupling
                     ):
        if min(coupling.sites) < 0 or max(coupling.sites) >= self.size_of_unit_cell:
            raise ValueError(f"Coupling sites {coupling.sites} are out of bounds for unit cell size {self.size_of_unit_cell}.")

        for i, existing in enumerate(self.couplings):
            if existing.sites == coupling.sites:
                self.couplings[i] = existing + coupling
                return

        self.couplings.append(coupling)



    def momentum_vec(self, *n:int):
        '''
        Compute the momentum vector in the Brillouin zone given a set of indices.
        Args:
            *n: A variable number of indices corresponding to the supercell vectors.
        Returns:
            A numpy array representing the momentum vector in the Brillouin zone.
        '''
        if len(n) != self.spatial_dim:
            raise ValueError(f"Expected {self.spatial_dim} indices, got {len(n)}.")
        if not all((0 <= ni < Ni) for ni, Ni in zip(n, self.Nvec)):
            raise ValueError(f"Indices must be in the Brillouin zone. The multiplicities are {self.Nvec}, but got indices {n}.")
        return np.array(n) @ self.bz_cellvectors
        
    # TODO: rewrite this in C++ and bind it to Python.
    def _build_adapted_basis(self,
                          Sztot:Optional[int] = None,
                          spin_flip:bool = False
                          )-> dict[tuple[int],int]:
        '''
        Build an adapted basis for the lattice, taking into account the symmetries of the system
        Args:
            Sztot: An optional integer representing the total number of particles. If provided, only states with this total number of particles will be included in the adapted basis.
        Returns:
            A dictionary mapping each unique state (as a tuple of integers) to its corresponding index in the adapted basis.
        '''

        def get_state(Sztot:Optional[int]) -> Iterator[tuple[int, ...]]:
            if Sztot is None:
                for state in product(range(self.physical_dim), repeat=self.size_of_unit_cell):
                    yield state
            else:
                for state in product(range(self.physical_dim), repeat=self.size_of_unit_cell):
                    if sum(state) == Sztot:
                        yield state
        d = {}
        for basis in get_state(Sztot):
            for sym in self.all_perms:
                transformed = tuple(sym(np.array(basis)))
                if transformed in d:
                    break
                if spin_flip:
                    flipped = tuple(self.physical_dim - 1 - np.array(transformed))
                    if flipped in d:
                        break
            else:
                d[tuple(basis)] = len(d)
            
        return d


    def set_center_at(self, center: int | np.ndarray):
        '''
        Center the lattice at a specific atom index or position.

        .. note::
            This only shifts stored positions; it does **not** rebuild
            ``all_perms`` or other derived quantities.  Call this before
            adding couplings or building the adapted basis.

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
        Accepts either a (d,d) matrix or a 0-d scalar (identity for any dimension).
        '''
        if arr.ndim == 0:
            # TrivialGroup identity: scalar 1 → acts as identity for any dimension
            pass
        else:
            assert arr.shape == (self.spatial_dim, self.spatial_dim), \
                f"Transformation matrix must be of shape ({self.spatial_dim}, {self.spatial_dim}), got {arr.shape}."

        permuted = np.zeros(self.size_of_unit_cell, dtype=int)

        for pos, s, idx in self.unit_cell_idx:
            transformed_pos = arr * pos if arr.ndim == 0 else arr @ pos
            permuted[idx] = self._get_idx(transformed_pos, s)
        return Permutation(permuted)
    
    def _get_permutation_tr(self, dir:int)-> Permutation:
        '''
        Given a translation vector dir, return the permutation of the atoms in the unit cell that corresponds to this translation.
        '''
        assert 0 <= dir < self.spatial_dim, f"Translation direction must be between 0 and {self.spatial_dim-1}."

        translated = np.zeros(self.size_of_unit_cell, dtype=int)
        for pos, s, idx in self.unit_cell_idx:
            translated[idx] = self._get_idx(pos + self.cell_vectors[dir], s)
        return Permutation(translated)

class Coupling:
    '''
    Class representing a coupling (interaction term) in the lattice.

    A ``Coupling`` stores one or more products of local operators acting on
    specific sites.  Each product is called a *term*.

    For example, the Heisenberg XX+YY+ZZ coupling on bond (0,1) is::

        Coupling(2, (0, 1), (X, X), (Y, Y), (Z, Z))

    Individual terms on the same sites can be added::

        Coupling(2, (0, 1), (X, X)) + Coupling(2, (0, 1), (Y, Y))
    '''

    def __init__(self,
                 phys_dim: int,
                 sites: tuple[int, ...],
                 *terms: tuple[np.ndarray, ...]):
        r'''
        Initialize a Coupling object.

        Args:
            phys_dim: The local Hilbert space dimension (e.g. 2 for spin-½).
            sites: Indices of the sites the operators act on.
            *terms: One or more tuples of operators, each of length
                    ``len(sites)``.  Every operator must be a
                    ``(phys_dim, phys_dim)`` array.

        Raises:
            ValueError: if no terms are given, or if the number of operators
                        in a term does not match the number of sites, or if
                        an operator has the wrong shape.
        '''
        if not terms:
            raise ValueError("At least one operator term is required.")

        for ops in terms:
            if len(ops) != len(sites):
                raise ValueError(
                    f"Each term must have {len(sites)} operators, got {len(ops)}."
                )
            for op in ops:
                if op.shape != (phys_dim, phys_dim):
                    raise ValueError(
                        f"All operators must have shape ({phys_dim}, {phys_dim}), "
                        f"got {op.shape}."
                    )

        self.phys_dim = phys_dim
        self.sites = sites
        self.terms: list[tuple[np.ndarray, ...]] = list(terms)

    # ── coupling_pair ─────────────────────────────────────────────────────

    def coupling_pair(self,
                      state: tuple[int, ...]
                      ) -> dict[tuple[int, ...], complex]:
        '''
        Apply the coupling to a basis state.

        For each term in the coupling, this computes every basis state
        reachable from *state* and the corresponding coefficient (matrix
        element).  Coefficients for the same final state are summed, so the
        result is always in simplest form.

        Args:
            state: A tuple of integers representing the initial basis state.
                   Its length must equal the total number of sites in the
                   system.

        Returns:
            A dict mapping each reachable basis state to its coefficient.

        Example:
            >>> X = np.array([[0, 1], [1, 0]])
            >>> Y = np.array([[0, -1j], [1j, 0]])
            >>> Z = np.array([[1, 0], [0, -1]])

            >>> zz = Coupling(2, (0, 1), (Z, Z))
            >>> zz.coupling_pair((0, 1))
            {(0, 1): (-1+0j)}

            >>> xx = Coupling(2, (0, 1), (X, X))
            >>> xx.coupling_pair((0, 1, 1)) # does not act on site 2
            {(1, 0, 1): (1+0j)}

            >>> heis = Coupling(2, (0, 1), (X, X), (Y, Y), (Z, Z))
            >>> heis.coupling_pair((0, 0))
            {(0, 0): (1+0j)}
        '''
        results: dict[tuple[int, ...], complex] = {}

        for ops in self.terms:
            per_site: list[list[tuple[int, complex]]] = []
            for site, op in zip(self.sites, ops):
                col = op[:, state[site]]
                nz = np.flatnonzero(col)
                per_site.append([(int(idx), complex(col[idx])) for idx in nz])

            for combo in product(*per_site):
                new_state = list(state)
                coeff = complex(1.0)
                for (new_val, amp), site in zip(combo, self.sites):
                    new_state[site] = new_val
                    coeff *= amp
                key = tuple(new_state)
                results[key] = results.get(key, 0j) + coeff

        return results

    def __add__(self, other: 'Coupling') -> 'Coupling':
        '''
        Add two couplings.  The two couplings must act on the same sites
        and have the same ``phys_dim``.
        '''
        if not isinstance(other, Coupling):
            return NotImplemented
        if self.sites != other.sites:
            raise ValueError(
                f"Cannot add couplings on different sites: "
                f"{self.sites} vs {other.sites}"
            )
        if self.phys_dim != other.phys_dim:
            raise ValueError(
                f"Cannot add couplings with different phys_dim: "
                f"{self.phys_dim} vs {other.phys_dim}"
            )
        return Coupling(self.phys_dim, self.sites, *self.terms, *other.terms)


    def __repr__(self) -> str:
        n_terms = len(self.terms)
        sites_str = str(self.sites)
        return (f"Coupling(phys_dim={self.phys_dim}, sites={sites_str}, "
                f"terms={n_terms})")


class Permutation:
    '''
    Class representing a symmetry operation on atoms in the lattice.
    '''
    def __init__(self, 
                 perm: np.ndarray):
        self.perm:NDArray[np.int64] = perm
        self.size = len(perm)
        if not np.array_equal(np.sort(perm), np.arange(self.size)):
            raise ValueError("Invalid permutation array. It must contain all integers from 0 to size-1 exactly once.")

    def __call__(self, x:np.ndarray|int) -> np.ndarray|int:
        if isinstance(x, int):
            return self.perm[x]
        elif isinstance(x, np.ndarray):
            return x[self.perm]

    def __mul__(self, other: 'Permutation') -> 'Permutation':
        if not isinstance(other, Permutation):
            raise TypeError("Can only multiply with another SymmetryOperation.")
        if self.size != other.size:
            raise ValueError("SymmetryOperations must be of the same size to multiply.")
        return Permutation(self.perm[other.perm])
    def inv(self)->Permutation:
        inv_perm = np.argsort(self.perm)
        return Permutation(inv_perm)
    def __pow__(self, power: int) -> 'Permutation':
        if power == 0:
            return Permutation(np.arange(self.size))
        elif power > 0:
            result = self
            for _ in range(power - 1):
                result = result * self
            return result
        else:
            return self.inv() ** (-power)
    def copy(self) -> Permutation:
        return Permutation(self.perm.copy())

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
    def all_irreps(self) -> list[str]:
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

    The only irrep of this group is 'A', which has a character of 1 for the identity operation.
    '''
    def __init__(self):
        self._iterated = False
        self._shape: tuple[int, ...] = ()
        self._operations: tuple[tuple[int | float, ...], ...] = ((1,),)

    def __hash__(self):
        return hash((self._shape, self._operations))

    def __eq__(self, other):
        if not isinstance(other, TrivialGroup):
            return NotImplemented
        return True

    def all_irreps(self) -> list[str]:
        return ['A']

    def little_group(self, k:np.ndarray,
                     reciprocal_basis:Optional[np.ndarray] = None
                     ) -> tuple['PointGroup', list[np.ndarray]]:
        return self, [np.eye(len(k))]

    def character(self, irrep:str, op:np.ndarray) -> complex:
        if irrep == 'A':
            return 1.0
        else:
            raise ValueError(f"Unknown irreducible representation: {irrep}")

    def __iter__(self):
        self._iterated = False
        return self

    def __next__(self) -> np.ndarray:
        if self._iterated:
            raise StopIteration
        self._iterated = True
        return np.array(self._operations[0], dtype=np.float64).reshape(self._shape)