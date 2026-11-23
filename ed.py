'''
Exact diagonalisation: the lattice model, its Hamiltonian and the sector
decomposition.

This is the consumer side of the symmetry machinery.  It owns the site table,
the couplings and the Hamiltonian, and it drives the three symmetry layers:

* ``representation`` gives the characters of the little group;
* ``lattice_action`` maps an operation onto a site permutation and onto the
  momentum labels;
* ``hilbert_action`` maps an operation onto a phased permutation of the
  computational basis.

Nothing in this module builds an operator on the full Hilbert space: every
symmetry operation is applied to a basis state or to a sparse vector, and every
projection is carried out on the momentum sector, whose dimension is
``dim H / |G_tr|``.
'''

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from base import PointGroup
from hilbert_action import Permutation
from lattice_action import TranslationGroup, little_group


# positions are compared after rounding to this many decimals; a symmetry
# operation maps a site onto another site only up to a supercell-lattice vector
_POS_TOL_DIGITS = 8
# a position is on a lattice site if its coordinates are integral to this
# tolerance (in units of the supercell vectors)
_POS_TOL = 1e-6


# key of a sector: (momentum label, little-group irrep, spin-flip parity)
SectorKey = tuple[tuple[int, ...], Optional[str], Optional[int]]


@dataclass
class Sector:
    '''
    One irreducible sector of the Hamiltonian.

    Attributes:
        key: ``(k, alpha, s)``, the momentum label, the little-group irrep
            (None when the point group is trivial) and the spin-flip parity
            (None when spin flip is not resolved).
        basis: ``(dim_H, dim_sector)`` sparse matrix whose orthonormal columns
            are the adapted basis vectors, in the computational basis.
        hamiltonian: ``(dim_sector, dim_sector)`` sparse Hamiltonian block in
            that basis.
    '''
    key: SectorKey
    basis: csr_matrix
    hamiltonian: csr_matrix

    @property
    def dim(self) -> int:
        '''Dimension of the sector.'''
        return int(self.basis.shape[1])


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

        # ── exact site lookup ──────────────────────────────────────────
        # a site is keyed by its fractional coordinates in the supercell basis,
        # reduced modulo one supercell vector: the same key is then obtained
        # whatever representative of the position a symmetry operation lands on
        self._site_lookup: dict[tuple[str, tuple[float, ...]], int] = {}
        for pos, s, idx in self.unit_cell_idx:
            key = (s, self._position_key(pos))
            if key in self._site_lookup:
                raise ValueError(f"Two sites of type {s!r} occupy the same position {pos}.")
            self._site_lookup[key] = idx

        # extending by PBC (used for plotting only; site lookup is exact):
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
                translated[idx] = self._site_index(pos + self.cell_vectors[i], s)
            self._elem_tr_perms.append(Permutation(np.argsort(translated)))

        # ── point-group permutations, validated as isometries of the torus ─
        self.point_group_ops: list[np.ndarray] = []
        self._pt_perms: list[Permutation] = []
        for arr in self.point_group:
            if arr.ndim != 0:
                assert arr.shape == (self.spatial_dim, self.spatial_dim), \
                    f"Transformation matrix must be of shape ({self.spatial_dim}, {self.spatial_dim}), got {arr.shape}."
            image = np.zeros(self.size_of_unit_cell, dtype=int)
            for pos, s, idx in self.unit_cell_idx:
                transformed_pos = arr * pos if arr.ndim == 0 else arr @ pos
                try:
                    image[idx] = self._site_index(transformed_pos, s)
                except ValueError:
                    raise ValueError(
                        f"Point-group operation {np.asarray(arr).tolist()} does not map the "
                        f"site {idx} at {np.asarray(pos).round(6).tolist()} onto a site of the "
                        f"supercell: it is not a symmetry of this lattice. Either enlarge the "
                        f"supercell or drop this operation.") from None
            if not np.array_equal(np.sort(image), np.arange(self.size_of_unit_cell)):
                raise ValueError(
                    f"Point-group operation {np.asarray(arr).tolist()} maps several sites onto "
                    f"the same site: it does not preserve the supercell lattice spanned by "
                    f"{self.supercell_vectors.tolist()}.")
            if tuple(np.argsort(image)) in {tuple(p.perm) for p in self._pt_perms}:
                continue    # duplicated operation
            self.point_group_ops.append(np.asarray(arr, dtype=float))
            self._pt_perms.append(Permutation(np.argsort(image)))

        # the permutation set must be closed, otherwise "G_pt" is not a group
        perms = {tuple(p.perm) for p in self._pt_perms}
        for a in self._pt_perms:
            for b in self._pt_perms:
                if tuple((a * b).perm) not in perms:
                    raise ValueError(
                        f"The supplied point group is not closed: the product of two of its "
                        f"operations is not in the set ({len(self._pt_perms)} distinct "
                        f"operations). Complete the group before building the model.")

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

    def _hilbert_space(self, Sztot: Optional[int] = None
                       ) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], int]]:
        '''
        Computational basis of the Hilbert space.

        Args:
            Sztot: If given, keep only states with ``sum(state) == Sztot``.
        Returns:
            (states, s2i): the basis states in lexicographic order and the map
            from a state to its row index.
        '''
        states = [s for s in product(range(self.physical_dim),
                                     repeat=self.size_of_unit_cell)
                  if Sztot is None or sum(s) == Sztot]
        return states, {s: i for i, s in enumerate(states)}

    def _translation_orbits(self, states: list[tuple[int, ...]]
                            ) -> list[tuple[int, ...]]:
        '''
        One representative (the smallest state) per translation orbit.

        Every orbit is walked once, so the cost is O(#orbits * |G_tr|) and not
        O(d^L * |G_tr|).
        '''
        seen: set[tuple[int, ...]] = set()
        reps: list[tuple[int, ...]] = []
        for state in states:
            if state in seen:
                continue
            orbit = {tuple(self._permutation(t, 0)(np.array(state)))
                     for t in self.tr_group}
            seen |= orbit
            reps.append(min(orbit))
        return reps

    def _build_H_sym(self, Sztot: Optional[int] = None) -> csr_matrix:
        '''
        The Hamiltonian, symmetrised over the full lattice group.

        ``H_sym = (1/|G_latt|) Σ_g U(g) H_raw U(g)^{-1}`` is the projector onto
        the operators that commute with the group: it returns ``H_raw``
        unchanged when the couplings already respect the symmetry, and the
        symmetric *average* when they do not.  Note that it averages rather
        than sums over the images, so a single local term is scaled down by
        1/|G_latt|; a Hamiltonian built from one term must be assembled from
        all of its images before being passed here.

        Args:
            Sztot: If given, work in the sector ``sum(state) == Sztot``.
        Returns:
            The symmetrised Hamiltonian as a complex CSR matrix.
        '''
        states, s2i = self._hilbert_space(Sztot)
        n = len(states)

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
        '''
        Momentum-sector Hamiltonians ``H_k = B_k^† H B_k``.  (Step 1 of 3.)

        Args:
            Sztot: If given, work in the sector ``sum(state) == Sztot``.
        Returns:
            A dict keyed by the momentum label ``k``; each value is the
            Hamiltonian in the orthonormal momentum-adapted basis of that
            sector.
        '''
        states, s2i = self._hilbert_space(Sztot)
        H_sym = self._build_H_sym(Sztot)
        result: dict[tuple[int, ...], csr_matrix] = {}
        for k in self.tr_group.all_momenta():
            reps, B, _ = self._momentum_basis(k, states, s2i)
            if reps:
                result[k] = (B.conj().T @ (H_sym @ B)).tocsr()
        return result

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

    def _momentum_basis(self, k: tuple[int, ...],
                        states: list[tuple[int, ...]],
                        s2i: dict[tuple[int, ...], int]
                        ) -> tuple[list[tuple[int, ...]], csr_matrix,
                                   dict[tuple[int, ...], int]]:
        '''
        Orthonormal basis of the momentum-k sector.

        Column x of the returned matrix is ``|k;x> = Π_k|x> / ||Π_k|x>||``,
        with x running over the translation-orbit representatives whose
        projection does not vanish.  Representatives of distinct orbits give
        orthogonal vectors, and ``Π_k(t x) = χ_k(t) Π_k x`` shows that a single
        representative per orbit is enough, so this set is an orthonormal basis
        of ``H_k``.

        Args:
            k: momentum label, an integer tuple.
            states: the computational basis.
            s2i: map from a state to its row index in *states*.
        Returns:
            (reps, B, state2col): the representatives, the
            ``(len(states), dim H_k)`` basis matrix, and the map from a state
            of the sector to the column of its orbit.
        '''
        reps: list[tuple[int, ...]] = []
        state2col: dict[tuple[int, ...], int] = {}
        rows: list[int] = []
        cols: list[int] = []
        data: list[complex] = []
        for rep in self._translation_orbits(states):
            vec = self.momentum_project(k, rep)
            norm = np.sqrt(sum(abs(c) ** 2 for c in vec.values()))
            if norm < 1e-12:
                continue        # k is incompatible with the stabiliser of |x>
            col = len(reps)
            reps.append(rep)
            for state, coeff in vec.items():
                state2col[state] = col
                rows.append(s2i[state])
                cols.append(col)
                data.append(coeff / norm)
        B = csr_matrix((data, (rows, cols)),
                       shape=(len(states), len(reps)), dtype=complex)
        return reps, B, state2col

    def _sector_action(self, state_map: Callable[[tuple[int, ...]], tuple[int, ...]],
                       states: list[tuple[int, ...]],
                       s2i: dict[tuple[int, ...], int],
                       state2col: dict[tuple[int, ...], int],
                       B: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
        '''
        Action of one group element on a momentum sector.

        Every momentum-adapted vector is mapped onto another one times a Bloch
        phase, ``U(g)|k;x> = φ[x] |k;x'>``: rotating a state that is delocalised
        over a translation orbit changes which orbit member has to be used as
        the representative, and that change of representative is a phase
        ``exp(-i k·t)``.  Dropping it makes the little-group projectors wrong
        and destroys the sector spectra, so the action is computed as
        ``B^† U(g) B`` and read off entry by entry.

        Args:
            state_map: the action of the group element on a computational basis
                state.
            states: the computational basis.
            s2i: map from a state to its row in *states*.
            state2col: map from a sector state to its column.
            B: ``(dim_H, dim_sector)`` orthonormal sector basis.
        Returns:
            (act, phase) with ``act[x]`` the column that ``U(g)`` maps column x
            to and ``phase[x]`` the coefficient of that column.
        Raises:
            ValueError: if the operation does not act inside the sector or if
                its action is not a phased permutation.
        '''
        coo = B.tocoo()
        # only the rows that carry the sector basis have to be mapped
        row_of: dict[int, int] = {}
        for i in np.unique(coo.row):
            image = state_map(states[i])
            if image not in s2i or image not in state2col:
                raise ValueError(
                    "The operation maps this momentum sector onto a different one, "
                    "so it is not in the little group of the momentum.")
            row_of[int(i)] = s2i[image]

        # U(g) B is B with every row moved to the row of its image
        moved = csr_matrix((coo.data, ([row_of[r] for r in coo.row], coo.col)),
                           shape=B.shape, dtype=complex)
        G = (B.conj().T @ moved).toarray()

        dim = B.shape[1]
        act = np.empty(dim, dtype=int)
        phase = np.empty(dim, dtype=complex)
        for x in range(dim):
            column = G[:, x]
            act[x] = int(np.argmax(np.abs(column)))
            phase[x] = column[act[x]]
            rest = np.linalg.norm(np.delete(column, act[x]))
            if rest > 1e-8 or abs(phase[x]) < 1e-8:
                raise ValueError(
                    "A group operation does not act on the momentum sector as a phased "
                    "permutation; check that the little group is correct.")
        return act, phase

    def little_group(self, k: tuple[int, ...]) -> tuple[PointGroup, list[np.ndarray]]:
        '''
        Little group of the momentum label *k*, plus one representative per coset.

        Args:
            k: momentum label, an integer tuple.
        Returns:
            (G_k, coset_reps): the subgroup of the point group that leaves *k*
            invariant, and a list of representatives of the *right* cosets
            ``G_k \\ G_pt`` (used only by the induced-character test).
        '''
        return little_group(self.point_group, k, self.supercell_vectors, self.Nvec)

    def _isotypic_basis(self, lg: PointGroup, alpha: str,
                        ops: list[np.ndarray],
                        actions: list[tuple[np.ndarray, np.ndarray]],
                        dim: int) -> np.ndarray:
        '''
        Orthonormal basis of the alpha-isotypic subspace of a momentum sector.

        ``Π_α = (d_α/|G_k|) Σ_r χ_α*(r) U(r)`` is applied to each basis vector
        and the images are orthogonalised one by one.  The images are in
        general neither orthogonal nor independent (that happens as soon as
        d_α > 1, and also when a basis vector is fixed by a non-trivial
        subgroup), so this orthonormalisation is what produces a basis.

        Args:
            lg: the little group.
            alpha: label of one of its irreps.
            ops: the operations of the little group, in the order of *actions*.
            actions: ``(act, phase)`` of each operation on the sector basis, as
                returned by ``_sector_action``.
            dim: dimension of the momentum sector.
        Returns:
            A complex ``(dim, m_alpha)`` matrix with orthonormal columns.
        '''
        d_alpha = lg.irrep_dim(alpha)
        # the projector is a sum of |G_k| phased permutations, i.e. cheap to
        # write down; its eigenvectors with eigenvalue one are the sector
        projector = np.zeros((dim, dim), dtype=complex)
        columns = np.arange(dim)
        for (a, phi), op in zip(actions, ops):
            projector[a, columns] += (np.conj(lg.character(alpha, op))
                                      * d_alpha / len(actions)) * phi
        eigenvalues, eigenvectors = np.linalg.eigh(projector)
        return eigenvectors[:, eigenvalues > 0.5]

    def _parity_basis(self, fmat: np.ndarray, s: int) -> np.ndarray:
        '''
        Orthonormal basis of the spin-flip parity ``s`` subspace of a sector.

        Args:
            fmat: ``(m, m)`` matrix of the spin flip in the sector basis.
            s: parity, +1 or -1.
        Returns:
            A complex ``(m, m_s)`` matrix with orthonormal columns, to be
            composed with the sector basis.
        '''
        projector = 0.5 * (np.eye(fmat.shape[0], dtype=complex) + s * fmat)
        eigenvalues, eigenvectors = np.linalg.eigh(projector)
        return eigenvectors[:, eigenvalues > 0.5]

    def build_sectors(self, Sztot: Optional[int] = None,
                      spin_flip: bool = False
                      ) -> dict[SectorKey, Sector]:
        '''
        Full symmetry-resolved decomposition of the Hamiltonian.

        For every momentum k, the little group G_k acts on the momentum-adapted
        basis by a permutation; projecting onto its irreps α and (optionally)
        onto the spin-flip parity s gives orthonormal bases of the sectors
        ``(k, α[, s])`` and the Hamiltonian blocks in those bases.

        Args:
            Sztot: If given, work in the sector ``sum(state) == Sztot``.
            spin_flip: If True, also resolve the spin-flip parity
                ``F: σ_i → d-1-σ_i``.
        Returns:
            A dict keyed by ``(k, alpha, s)`` — ``alpha`` is None when the point
            group is trivial and ``s`` is None when *spin_flip* is False — of
            :class:`Sector` objects holding the basis and the block Hamiltonian.
        '''
        if spin_flip and Sztot is not None and \
                2 * Sztot != self.size_of_unit_cell * (self.physical_dim - 1):
            raise ValueError(
                "The spin flip maps sum(state) == Sztot onto "
                "L(d-1) - Sztot; it is a symmetry of a single sector only when "
                "Sztot is self-conjugate.")
        states, s2i = self._hilbert_space(Sztot)
        H_sym = self._build_H_sym(Sztot)
        flip = (lambda state: tuple(self.physical_dim - 1 - s for s in state))
        sectors: dict[SectorKey, Sector] = {}
        for k in self.tr_group.all_momenta():
            reps, B, state2col = self._momentum_basis(k, states, s2i)
            if not reps:
                continue
            H_k = (B.conj().T @ (H_sym @ B)).toarray()
            lg, _ = self.little_group(k)
            ops = list(lg)
            fact = (self._sector_action(flip, states, s2i, state2col, B)
                    if spin_flip else None)
            # the trivial little group has a single sector, nothing to project
            actions = [] if len(ops) == 1 else [
                self._sector_action(
                    (lambda state, perm=self._op_perm(op):
                     tuple(perm(np.array(state)))),
                    states, s2i, state2col, B)
                for op in ops]
            irreps = ['A'] if len(ops) == 1 else lg.all_irrps()
            for alpha in irreps:
                Q = (np.eye(len(reps), dtype=complex) if len(ops) == 1
                     else self._isotypic_basis(lg, alpha, ops, actions, len(reps)))
                if Q.shape[1] == 0:
                    continue
                H_a = Q.conj().T @ (H_k @ Q)
                # the spin flip commutes with the whole symmetry group, so it
                # acts inside the alpha sector; its matrix there follows from
                # its (phased) permutation of the sector basis
                fmat = None
                if spin_flip:
                    pfq = np.zeros_like(Q)
                    pfq[fact[0], :] = fact[1][:, None] * Q
                    fmat = Q.conj().T @ pfq
                for s in ((None,) if not spin_flip else (1, -1)):
                    R = (np.eye(Q.shape[1], dtype=complex) if s is None
                         else self._parity_basis(fmat, s))
                    if R.shape[1] == 0:
                        continue
                    key = (k, alpha, s)
                    sectors[key] = Sector(
                        key=key, basis=csr_matrix(B @ (Q @ R)),
                        hamiltonian=csr_matrix(R.conj().T @ (H_a @ R)))
        return sectors

    def _op_perm(self, op: np.ndarray) -> Permutation:
        '''
        Site permutation of a point-group operation.

        The trivial group is represented by a scalar, which stands for the
        identity permutation.

        Args:
            op: an operation of ``self.point_group``.
        Returns:
            The corresponding site permutation.
        '''
        arr = np.asarray(op, dtype=float)
        if arr.ndim == 0:
            return Permutation(np.arange(self.size_of_unit_cell))
        return self._pt_perms[self._pt_index(arr)]

    def _pt_index(self, op: np.ndarray) -> int:
        '''Index of the point-group operation *op* in ``self._pt_perms``.'''
        for i, other in enumerate(self.point_group_ops):
            if np.allclose(other, op):
                return i
        raise ValueError(f"Operation {np.asarray(op).tolist()} not found in the point group.")

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

    def _site_index(self, pos: np.ndarray, s: str) -> int:
        '''
        Index of the site of type *s* sitting at *pos*, modulo the supercell lattice.

        The position is reduced modulo ``supercell_vectors`` before the lookup,
        so a symmetry operation may land on any periodic image of a site.

        Args:
            pos: Cartesian position of a site, up to a supercell-lattice vector.
            s: site type.

        Raises:
            ValueError: if no site of type *s* sits at that position.
        '''
        key = (s, self._position_key(pos))
        if key not in self._site_lookup:
            raise ValueError(f"No site of type {s!r} at {np.asarray(pos).round(6).tolist()}.")
        return self._site_lookup[key]

    def _position_key(self, pos: np.ndarray) -> tuple[float, ...]:
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

    def _get_idx(self, pos: np.ndarray, s: str) -> int:
        '''Backward-compatible alias of :meth:`_site_index`.'''
        return self._site_index(pos, s)


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
