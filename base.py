from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from typing import Optional, Iterator

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


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

        # ── translation group ──────────────────────────────────────────
        self.tr_group = TranslationGroup(
            tuple(int(n) for n in self.Nvec), self.cell_vectors)

        # ── elementary permutations (d + |G_pt| stored, not the product) ─
        self._elem_tr_perms: list[Permutation] = []
        for i in range(self.spatial_dim):
            translated = np.zeros(self.size_of_unit_cell, dtype=int)
            for pos, s, idx in self.unit_cell_idx:
                translated[idx] = self._get_idx(pos + self.cell_vectors[i], s)
            self._elem_tr_perms.append(Permutation(translated))

        self._pt_perms: list[Permutation] = []
        for arr in self.point_group:
            if arr.ndim != 0:
                assert arr.shape == (self.spatial_dim, self.spatial_dim), \
                    f"Transformation matrix must be of shape ({self.spatial_dim}, {self.spatial_dim}), got {arr.shape}."
            permuted = np.zeros(self.size_of_unit_cell, dtype=int)
            for pos, s, idx in self.unit_cell_idx:
                transformed_pos = arr * pos if arr.ndim == 0 else arr @ pos
                permuted[idx] = self._get_idx(transformed_pos, s)
            self._pt_perms.append(Permutation(permuted))

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



    # ── Hamiltonian construction ────────────────────────────────────────────

    def _tr_representatives(self, Sztot: Optional[int] = None
                            ) -> list[tuple[int, ...]]:
        '''Representatives of translation-only orbits.'''
        seen = set()
        reps = []
        for state in product(range(self.physical_dim), repeat=self.size_of_unit_cell):
            if Sztot is not None and sum(state) != Sztot:
                continue
            # Orbit of *state* under translations — take smallest as representative
            rep = min(tuple(self._permutation(t, 0)(np.array(state)))
                      for t in self.tr_group)
            if rep not in seen:
                seen.add(rep)
                reps.append(rep)
        return reps

    def _build_H_sym(self, Sztot: Optional[int] = None) -> csr_matrix:
        states = [s for s in product(range(self.physical_dim),
                                       repeat=self.size_of_unit_cell)
                  if Sztot is None or sum(s) == Sztot]
        n = len(states)
        s2i = {s: i for i, s in enumerate(states)}

        # build H_raw as sparse (COO accumulation)
        raw_acc: dict[tuple[int, int], complex] = defaultdict(complex)
        for i, s in enumerate(states):
            for c in self.couplings:
                for f, coeff in c.coupling_pair(s).items():
                    j = s2i.get(f)
                    if j is not None:
                        raw_acc[(i, j)] += coeff

        # symmetrize: accumulate (1/|G|) Σ_g g H_raw g^{-1} into sparse dict
        n_g = len(self.tr_group) * len(self._pt_perms)
        sym_acc: dict[tuple[int, int], complex] = defaultdict(complex)

        # precompute forward permutation for each group element
        for t in self.tr_group:
            for r_idx in range(len(self._pt_perms)):
                perm = self._permutation(t, r_idx)
                pfwd = np.zeros(n, dtype=int)
                for i, s in enumerate(states):
                    gs = tuple(perm(np.array(s)))
                    pfwd[i] = s2i[gs]
                # U(g) H_raw U(g)^{-1} has matrix element at (pfwd[i], pfwd[j])
                for (i, j), v in raw_acc.items():
                    sym_acc[(pfwd[i], pfwd[j])] += v

        data, rows, cols = [], [], []
        inv_n = 1.0 / n_g
        for (i, j), v in sym_acc.items():
            v_scaled = v * inv_n
            if abs(v_scaled) > 1e-12:
                data.append(v_scaled)
                rows.append(i)
                cols.append(j)
        return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=complex)

    def _build_H(self, Sztot: Optional[int] = None
                 ) -> dict[tuple[int, ...], csr_matrix]:
        '''Build H_k for each momentum sector k.  (Step 1 of 3.)'''
        states = [s for s in product(range(self.physical_dim),
                                       repeat=self.size_of_unit_cell)
                  if Sztot is None or sum(s) == Sztot]
        s2i = {s: i for i, s in enumerate(states)}
        rep_list = self._tr_representatives(Sztot)
        Gtr = len(self.tr_group)
        H_sym = self._build_H_sym(Sztot)

        result = {}
        for k in self.tr_group.all_momenta():
            k_reps, k_norms = [], []
            for rep in rep_list:
                phase_sum = sum(np.conj(self.tr_group.character(k, t))
                               for t in self.tr_group
                               if tuple(self._permutation(t, 0)(np.array(rep))) == rep)
                norm_sq = abs(phase_sum) / Gtr
                if norm_sq > 1e-12:
                    k_reps.append(rep)
                    k_norms.append(np.sqrt(norm_sq))
            Mk = len(k_reps)
            if Mk == 0:
                continue

            Hk_data, Hk_rows, Hk_cols = [], [], []
            for i in range(Mk):
                idx_i = s2i[k_reps[i]]
                ni = 1.0 / (k_norms[i] * Gtr)
                for j in range(Mk):
                    rep_j = k_reps[j]
                    nj = 1.0 / (k_norms[j] * Gtr)
                    elem = 0.0j
                    for t in self.tr_group:
                        tx = tuple(self._permutation(t, 0)(np.array(rep_j)))
                        h_val = H_sym[idx_i, s2i[tx]]
                        if h_val != 0:
                            elem += np.conj(self.tr_group.character(k, t)) * h_val
                    elem *= ni * nj * Gtr
                    if abs(elem) > 1e-12:
                        Hk_data.append(elem); Hk_rows.append(i); Hk_cols.append(j)
            H_k = csr_matrix((Hk_data, (Hk_rows, Hk_cols)), shape=(Mk, Mk),
                             dtype=complex)
            result[k] = H_k

        return result


    # TODO: rewrite this in C++ and bind it to Python.
    def _build_adapted_basis(self,
                             Sztot:Optional[int] = None,
                             spin_flip:bool = False
                             )-> dict[tuple[int],int]:
        '''
        Build an adapted basis for the lattice, taking into account the symmetries of the system
        Args:
            Sztot: An optional integer representing the total number of particles. If provided, only states with this total number of particles will be included in the adapted basis.
            spin_flip: If True, also include states obtained by spin-flip symmetry.
        Returns:
            A dictionary mapping each unique state (as a tuple of integers) to its corresponding index in the adapted basis.
        '''
        d = {}
        for basis in product(range(self.physical_dim), repeat=self.size_of_unit_cell):
            if Sztot is not None and sum(basis) != Sztot:
                continue
            orb = self.orbit(basis, spin_flip=spin_flip)
            if orb[0] not in d:
                d[orb[0]] = len(d)
        return d

    # ── Permutation composition ───────────────────────────────────────────

    def _permutation(self, t: tuple[int, ...], r_idx: int) -> Permutation:
        '''Compose  φ(t, r) = φ_tr(t) ∘ φ_pt(r)  on the fly.'''
        n_sites = self.size_of_unit_cell
        tr_perm = Permutation(np.arange(n_sites))
        for i, ti in enumerate(t):
            if ti:
                tr_perm = tr_perm * (self._elem_tr_perms[i] ** ti)
        return tr_perm * self._pt_perms[r_idx]

    # ── Orbit ─────────────────────────────────────────────────────────────

    def orbit(self, state: tuple[int, ...],
              spin_flip: bool = False) -> list[tuple[int, ...]]:
        '''
        Compute the orbit of a basis state under the full lattice group
        G_latt = G_tr ⋊ G_pt.

        Args:
            state: A tuple of integers representing the initial basis state.
            spin_flip: If True, also include spin-flipped states.
        Returns:
            A sorted list of unique tuples.
        '''
        orb = []
        for t in self.tr_group:
            for r_idx in range(len(self._pt_perms)):
                perm = self._permutation(t, r_idx)
                orb.append(tuple(perm(np.array(state))))
        if spin_flip:
            flipped = tuple(self.physical_dim - 1 - np.array(state))
            for t in self.tr_group:
                for r_idx in range(len(self._pt_perms)):
                    perm = self._permutation(t, r_idx)
                    orb.append(tuple(perm(np.array(flipped))))
        return sorted(set(orb))

    # ── Two-step projection ────────────────────────────────────────────────

    def momentum_project(self, k: tuple[int, ...],
                         state: tuple[int, ...]
                         ) -> dict[tuple[int, ...], complex]:
        '''
        Step 1: project onto momentum sector k using translations only.

            Π_k |x⟩ = (1/|G_tr|) Σ_{t∈G_tr} exp(-i k·t) · t|x⟩

        Returns a dict mapping projected basis states to coefficients.
        '''
        result: dict[tuple[int, ...], complex] = {}
        inv_norm = 1.0 / len(self.tr_group)
        for t in self.tr_group:
            perm = self._permutation(t, 0)  # r_idx=0 → identity rotation
            coeff = inv_norm * np.conj(self.tr_group.character(k, t))
            projected = tuple(perm(np.array(state)))
            result[projected] = result.get(projected, 0j) + coeff
        return result

    def little_group_project(self, k: tuple[int, ...],
                             alpha: str,
                             state: tuple[int, ...]
                             ) -> dict[tuple[int, ...], complex]:
        '''
        Step 2: within momentum sector k, project onto little-group irrep α.

            Π_α^{G_k} = (d_α/|G_k|) Σ_{r∈G_k} χ_α*(r) · r

        *state* should already be momentum-projected (output of
        ``momentum_project``).  Returns a dict of coefficients.
        '''
        k_vec = np.array(k) @ self.bz_cellvectors
        lg, _ = self.point_group.little_group(k_vec, self.bz_cellvectors)
        lg_ops = list(lg)
        d_alpha = lg.character(alpha, next(iter(lg)))
        inv_norm = d_alpha / len(lg_ops)

        result: dict[tuple[int, ...], complex] = {}
        for r_op in lg_ops:
            # find index of r_op in self._pt_perms
            for i, r in enumerate(self.point_group):
                if np.allclose(r, r_op):
                    r_idx = i
                    break
            perm = self._pt_perms[r_idx]
            coeff = inv_norm * np.conj(lg.character(alpha, r_op))
            projected = tuple(perm(np.array(state)))
            result[projected] = result.get(projected, 0j) + coeff
        return result

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

    def _get_idx(self, pos: np.ndarray, s: str, tol: float = 1e-6) -> int:
        # brute force search
        # self.extended_cell is at most 1000, so this is fine
        for p, s1, idx in self.extended_cell:
            if np.allclose(p, pos, atol=tol) and s1 == s:
                return idx
        raise ValueError(f"Position {pos} not found in unit cell or extended cell.")

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
        '''
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
            raise TypeError("Can only multiply with another Permutation.")
        if self.size != other.size:
            raise ValueError("Permutations must be of the same size to multiply.")
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

    def all_irrps(self) -> list[str]:
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


