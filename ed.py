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

The decomposition itself is the kernel's: ``build_sectors`` hands the model to
``kernel`` and gets sparse blocks back.  Nothing here loops over states or over
the columns of a sector, and nothing here builds an operator on the full Hilbert
space.
'''

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

import kernel

from base import LatticeAction, PointGroup
from hilbert_action import MonomialAction, Permutation, extract_monomial, from_lattice_action
from lattice_action import AffineAction, Lattice, TranslationGroup, little_group
from representation import TrivialGroup


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
                 MAX_EXTENSION = 1,
                 orbital_mixing: Optional[dict[str, Callable[[np.ndarray], np.ndarray]]] = None,
                 local_charge: Optional[tuple[int, ...]] = None
                 ):
        '''
        Initialize a lattice model.

        Args:
            phys_dim: dimension of the local Hilbert space of one site.
            supercell_info: one ``(vector, multiplicity)`` pair per direction;
                the supercell vector is ``vector * multiplicity``.
            cell_info: position of every site inside the cell, keyed by
                ``(species, label)``.  A symmetry operation may exchange sites
                of the same species -- the two sublattices of a honeycomb have
                to share their species for the sixfold rotation to be a
                symmetry -- but never sites of different species.  The label is
                only there to make the key unique.
            point_group: the point group of the lattice; a trivial group when
                None.  An element is either a bare ``(d, d)`` rotation about the
                origin, or a full ``LatticeAction`` -- the latter is how a glide
                plane or a screw axis enters, since those are a rotation plus a
                fractional translation.
            MAX_EXTENSION: half-width of the periodically extended copy kept
                for plotting.
            orbital_mixing: per species, the local action of a rotation on that
                species' orbitals, as a function ``rotation -> (d, d) unitary``.
                The matrix has to be monomial -- one non-zero per column -- so
                that an operation still maps one basis state onto one basis
                state; in two dimensions the circular harmonics ``(x ± iy)^m``
                (see ``circle_doublet``) always are.  None means every orbital
                is a scalar and the operation only permutes sites.
            local_charge: charge of every local basis state, used to select the
                sector ``sum(charge) == Sztot``; defaults to ``0, 1, ..., d-1``,
                which is the spin or particle number when the local space is a
                spin degeneracy.
        Raises:
            ValueError: if the point group contains an operation that is not a
                symmetry of this finite lattice -- which includes any operation
                that would exchange sites of different species -- if two of its
                operations act as the same site permutation, or if it is not
                closed.
        '''
        self.physical_dim = phys_dim
        self.local_charge: tuple[int, ...] = (tuple(range(phys_dim)) if local_charge is None
                                              else tuple(int(c) for c in local_charge))
        if len(self.local_charge) != phys_dim:
            raise ValueError(f"local_charge must have one entry per local state ({phys_dim}).")
        self.lattice = Lattice(supercell_info, cell_info, MAX_EXTENSION)
        self.spatial_dim = self.lattice.spatial_dim
        self.point_group = TrivialGroup() if point_group is None else point_group

        # ── translation group and its elementary site permutations ──────
        self.tr_group = TranslationGroup(tuple(int(n) for n in self.lattice.Nvec),
                                         self.lattice.cell_vectors)
        self._elem_tr_perms: list[Permutation] = [
            Permutation(np.argsort(np.array([
                self.lattice.site_index(pos + self.lattice.cell_vectors[i], s)
                for pos, s, _ in self.lattice.unit_cell_idx])))
            for i in range(self.spatial_dim)]

        # ── local action: how a rotation acts on every species' orbitals ──
        self.species: list[str] = [s for _, s, _ in self.lattice.unit_cell_idx]
        known = set(self.species)
        if orbital_mixing is not None:
            unknown = set(orbital_mixing) - known
            if unknown:
                raise ValueError(f"orbital_mixing names species that do not occur: {sorted(unknown)}.")

        # ── point-group actions: geometry -> permutation of the basis ───
        self.actions: list[AffineAction] = []
        self.site_perms: list[Permutation] = []
        self.local: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
        for op in self.point_group:
            # an element may be a bare matrix (a rotation about the origin) or
            # a full affine action, which is how a glide or a screw enters
            action = (op if isinstance(op, LatticeAction)
                      else AffineAction.from_matrix(op, self.spatial_dim))
            perm = from_lattice_action(action, self.lattice)   # rejects non-symmetries
            local = {sp: extract_monomial(mixing(action.rotation))
                     for sp, mixing in (orbital_mixing or {}).items()}
            # two operations are the same element only if both the site
            # permutation and the local action agree: on a small torus a
            # rotation and a translation can move the sites identically while
            # acting differently on the orbitals
            def signature(perm, local):
                return (tuple(perm.perm),
                        tuple(sorted((sp, tuple(np.round(p, 8)), tuple(np.round(ph, 8)))
                                     for sp, (p, ph) in local.items())))
            if signature(perm, local) in {signature(p, l)
                                         for p, l in zip(self.site_perms, self.local)}:
                continue        # duplicated operation
            self.actions.append(action)
            self.site_perms.append(perm)
            self.local.append(local)

        self.hilbert: list[MonomialAction] = [
            MonomialAction(perm, self.species, local)
            for perm, local in zip(self.site_perms, self.local)]

        # the site permutations must be closed, otherwise this is not a group
        perms = {tuple(p.perm) for p in self.site_perms}
        for a in self.site_perms:
            for b in self.site_perms:
                if tuple((a * b).perm) not in perms:
                    raise ValueError(
                        f"The supplied point group is not closed: the product of two of its "
                        f"operations is not in the set ({len(self.site_perms)} distinct "
                        f"operations). Complete the group before building the model.")

        self.couplings: list[Coupling] = []

    # ── geometry facade: the data lives on self.lattice ─────────────────
    # these are delegations, not copies, so there is one source of truth

    @property
    def size_of_unit_cell(self) -> int:
        '''Number of sites in the supercell.'''
        return self.lattice.size_of_unit_cell

    @property
    def unit_cell_idx(self) -> list[tuple[np.ndarray, str, int]]:
        '''``(position, type, index)`` of every site.'''
        return self.lattice.unit_cell_idx

    @property
    def cell_vectors(self) -> NDArray[np.float64]:
        '''Rows are the unit cell vectors.'''
        return self.lattice.cell_vectors

    @property
    def Nvec(self) -> NDArray[np.int64]:
        '''Multiplicities of the unit cell vectors.'''
        return self.lattice.Nvec

    @property
    def supercell_vectors(self) -> NDArray[np.float64]:
        '''Rows are the supercell vectors.'''
        return self.lattice.supercell_vectors

    @property
    def bz_cellvectors(self) -> NDArray[np.float64]:
        '''Rows are the reciprocal vectors of the supercell.'''
        return self.lattice.bz_cellvectors



    def add_coupling(self,
                     coupling: Coupling
                     ):
        if min(coupling.sites) < 0 or max(coupling.sites) >= self.lattice.size_of_unit_cell:
            raise ValueError(f"Coupling sites {coupling.sites} are out of bounds for unit cell size {self.lattice.size_of_unit_cell}.")

        for i, existing in enumerate(self.couplings):
            if existing.sites == coupling.sites:
                self.couplings[i] = existing + coupling
                return

        self.couplings.append(coupling)



    # ── Hamiltonian construction ────────────────────────────────────────────

    def _translation_perm(self, t: tuple[int, ...]) -> Permutation:
        '''Site permutation of the translation *t*.'''
        perm = Permutation(np.arange(self.lattice.size_of_unit_cell))
        for i, ti in enumerate(t):
            if ti:
                perm = perm * (self._elem_tr_perms[i] ** ti)
        return perm

    def _permutation(self, t: tuple[int, ...], r_idx: int) -> Permutation:
        '''Compose  φ(t, r) = φ_tr(t) ∘ φ_pt(r)  on the fly.'''
        n_sites = self.lattice.size_of_unit_cell
        tr_perm = Permutation(np.arange(n_sites))
        for i, ti in enumerate(t):
            if ti:
                tr_perm = tr_perm * (self._elem_tr_perms[i] ** ti)
        return tr_perm * self.site_perms[r_idx]

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
            for r_idx in range(len(self.hilbert)):
                perm = self._permutation(t, r_idx)
                orb.append(tuple(perm(np.array(state))))
        if spin_flip:
            flipped = tuple(self.physical_dim - 1 - np.array(state))
            for t in self.tr_group:
                for r_idx in range(len(self.hilbert)):
                    perm = self._permutation(t, r_idx)
                    orb.append(tuple(perm(np.array(flipped))))
        return sorted(set(orb))

    # ── Two-step projection ────────────────────────────────────────────────

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
        return little_group(self.point_group, k, self.lattice.supercell_vectors, self.lattice.Nvec)

    def _spin_flip_action(self) -> MonomialAction:
        '''
        The spin flip on the computational basis: ``σ_i -> d-1-σ_i`` on every site.

        It is a local action only -- the sites do not move -- so it is the same
        monomial action for every species.

        Returns:
            The monomial action of the spin flip.
        '''
        local = {sp: (np.array([self.physical_dim - 1 - i
                                for i in range(self.physical_dim)]),
                      np.ones(self.physical_dim, dtype=complex))
                 for sp in set(self.species)}
        return MonomialAction(Permutation(np.arange(self.lattice.size_of_unit_cell)),
                              self.species, local)

    def build_sectors(self, Sztot: Optional[int] = None,
                      spin_flip: bool = False
                      ) -> dict[SectorKey, Sector]:
        '''
        Full symmetry-resolved decomposition of the Hamiltonian.

        For every momentum k the little group G_k acts on the momentum-adapted
        basis by phased permutations; projecting onto its irreps α and
        (optionally) onto the spin-flip parity s gives orthonormal bases of the
        sectors ``(k, α[, s])`` and the Hamiltonian blocks in those bases.  The
        work is done by the kernel through :mod:`kernel`; everything that
        decides physics -- the little groups, the characters, allowed or
        forbidden operations -- is settled here before the kernel is called.

        Args:
            Sztot: If given, work in the sector ``sum(state) == Sztot``.
            spin_flip: If True, also resolve the spin-flip parity
                ``F: σ_i → d-1-σ_i``.
        Returns:
            A dict keyed by ``(k, alpha, s)`` — ``alpha`` is ``'A'`` when the
            little group is trivial and ``s`` is None when *spin_flip* is False —
            of :class:`Sector` objects holding the basis and the block
            Hamiltonian, both sparse.
        Raises:
            ValueError: if *spin_flip* is asked for a charge sector that the
                spin flip does not map onto itself.
        '''
        if spin_flip and Sztot is not None and \
                2 * Sztot != self.lattice.size_of_unit_cell * (self.physical_dim - 1):
            raise ValueError(
                "The spin flip maps sum(state) == Sztot onto "
                "L(d-1) - Sztot; it is a symmetry of a single sector only when "
                "Sztot is self-conjugate.")
        self.check_symmetry()
        built = kernel.sectors(self, Sztot, spin_flip)
        return {key: Sector(key=key, basis=basis, hamiltonian=block)
                for key, (basis, block) in built.items()}

    def spectra(self, Sztot: Optional[int] = None, count: int = 5,
                spin_flip: bool = False, threads: int = 0
                ) -> dict[SectorKey, NDArray[np.float64]]:
        '''
        Lowest levels of every sector, built and solved one sector at a time.

        This is the method to use when what is wanted is the spectrum: it never
        holds more than one sector's Hamiltonian and one block at a time, and it
        parallelizes the construction of that one sector rather than the set of
        them.

        Args:
            Sztot: If given, work in the sector ``sum(state) == Sztot``.
            count: how many levels per sector.
            spin_flip: If True, also resolve the spin-flip parity.
            threads: workers for the sector Hamiltonian; 0 means one per core.
        Returns:
            A dict keyed by ``(k, alpha, s)`` of the eigenvalues, ascending.
        Raises:
            ValueError: if *spin_flip* is asked for a charge sector the spin flip
                does not preserve, or if the couplings are not closed under the
                group.
        '''
        if spin_flip and Sztot is not None and \
                2 * Sztot != self.lattice.size_of_unit_cell * (self.physical_dim - 1):
            raise ValueError(
                "The spin flip maps sum(state) == Sztot onto "
                "L(d-1) - Sztot; it is a symmetry of a single sector only when "
                "Sztot is self-conjugate.")
        self.check_symmetry()
        return kernel.spectra(self, count=count, sztot=Sztot, spin_flip=spin_flip,
                              threads=threads)

    def spectra_free(self, Sztot: Optional[int] = None, count: int = 5,
                     threads: int = 0) -> dict[tuple[int, ...], NDArray[np.float64]]:
        '''
        Lowest levels of every momentum, without ever forming a matrix.

        Args:
            Sztot: If given, work in the sector ``sum(state) == Sztot``.
            count: how many levels per momentum.
            threads: workers for the product; 0 means one per core.
        Returns:
            A dict keyed by the momentum label of the eigenvalues, ascending.
            The little group is not resolved here: the momentum sector is one
            block, and the point-group irreps are what `spectra` adds on top.
        Raises:
            ValueError: if the couplings are not closed under the space group.
        '''
        self.check_symmetry()
        return kernel.spectra_free(self, count=count, sztot=Sztot, threads=threads)

    def check_symmetry(self) -> None:
        '''
        Check that the couplings are closed under the space group.

        The decomposition projects the Hamiltonian onto the sectors of the group,
        which is only meaningful when the Hamiltonian commutes with the group.
        The couplings are the Hamiltonian, so what has to hold is that the image
        of every coupling is again a coupling of the model -- with the same local
        operators, which is not checked here: the site pattern is what the
        decomposition depends on, and it is what goes wrong in practice (a bond
        list that looks periodic but misses the bonds that close the torus).

        Raises:
            ValueError: if the image of a coupling is not a coupling.
        '''
        known = {frozenset(coupling.sites) for coupling in self.couplings}
        for t in self.tr_group:
            translation = self._translation_perm(t)
            for index in range(len(self.hilbert)):
                action = self.hilbert[index].after_translation(translation)
                # the stored permutation is the preimage map, so the image of a
                # site is where that map sends its value
                image = np.argsort(action.site.perm)
                for coupling in self.couplings:
                    moved = frozenset(int(image[site]) for site in coupling.sites)
                    if moved not in known:
                        raise ValueError(
                            "The couplings are not invariant under the space group: the "
                            f"coupling on sites {sorted(coupling.sites)} is mapped onto "
                            f"{sorted(moved)}, which is not a coupling of the model. "
                            "The sector decomposition would project a Hamiltonian that "
                            "does not commute with the group.")

    def _action_for(self, op: np.ndarray) -> MonomialAction:
        '''
        Action on the computational basis of a point-group operation.

        The trivial group is represented by a scalar, which stands for the
        identity.

        Args:
            op: an operation of ``self.point_group``.
        Returns:
            The corresponding monomial action.
        Raises:
            ValueError: if the operation is not part of the model's group.
        '''
        arr = np.asarray(op, dtype=float)
        if arr.ndim == 0:
            return MonomialAction(Permutation(np.arange(self.lattice.size_of_unit_cell)),
                                  self.species, {})
        for action, hilbert in zip(self.actions, self.hilbert):
            if np.allclose(action.rotation, arr):
                return hilbert
        raise ValueError(f"Operation {arr.tolist()} not found in the point group.")

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
