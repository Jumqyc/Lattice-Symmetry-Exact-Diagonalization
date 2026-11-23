'''
The bridge to the compiled kernel.

The group theory, the model description and the API stay in Python; the loops
over states, columns and group elements are in ``_edkern``.  This module is the
only place that knows both sides: it turns a :class:`ed.Model` into the plain
arrays the kernel takes and turns the kernel's sparse results back into
``scipy`` matrices.

Nothing here decides physics.  The little groups, the characters and the spin
flip all come from the Python layer; the kernel is asked only for objects it can
build from the arrays it is given.
'''

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy.sparse as sp

try:
    import _edkern
except ImportError:                     # pragma: no cover - the kernel is optional
    _edkern = None

if TYPE_CHECKING:                        # pragma: no cover
    from ed import Model


def require() -> None:
    '''
    Raise if the kernel is missing, with the command that builds it.

    Raises:
        RuntimeError: if ``_edkern`` cannot be imported.
    '''
    if _edkern is None:
        raise RuntimeError(
            "the compiled kernel is not available: build it with "
            "'python cpp/build.py'")


# ── the model, as the kernel's arrays ─────────────────────────────────────────

def basis_of(model: 'Model', sztot: Optional[int]):
    '''
    The computational basis of a charge sector.

    Args:
        model: the model.
        sztot: total charge, or None for the whole space.
    Returns:
        The kernel's ``Basis``.
    '''
    require()
    return _edkern.Basis(model.size_of_unit_cell, model.physical_dim,
                         list(model.local_charge), -1 if sztot is None else sztot)


def coupling_terms(model: 'Model') -> list:
    '''
    The Hamiltonian's terms, one ``CouplingTerm`` per (coupling, product).

    Args:
        model: the model.
    Returns:
        The terms in the kernel's representation.  A coupling that carries
        several products contributes several terms: concatenating them into one
        local table would mix them into a different operator.
    '''
    require()
    terms = []
    for coupling in model.couplings:
        for term in coupling.terms:
            local = []
            for position in range(len(coupling.sites)):
                operator = term[position]
                outs, ins, coeffs = [], [], []
                for column in range(operator.shape[1]):
                    for row in range(operator.shape[0]):
                        if abs(operator[row, column]) <= 1e-14:
                            continue
                        outs.append(row)
                        ins.append(column)
                        coeffs.append(complex(operator[row, column]))
                local.append(_edkern.LocalTerm(outs, ins, coeffs))
            terms.append(_edkern.CouplingTerm(list(coupling.sites), local))
    return terms


def monomial_of(model: 'Model', action) -> '_edkern.MonomialAction':
    '''
    A Hilbert action in the kernel's representation.

    Args:
        model: the model.
        action: a :class:`hilbert_action.HilbertAction`.
    Returns:
        The site permutation, and the local permutation and phase of every
        site and local state.  A species without a local part gets the identity.
    '''
    require()
    sites, dim = model.size_of_unit_cell, model.physical_dim
    perm = np.tile(np.arange(dim), sites)
    phase = np.ones(sites * dim, dtype=complex)
    for site, species in enumerate(model.species):
        if species not in action.local:
            continue
        local_perm, local_phase = action.local[species]
        perm[site * dim:(site + 1) * dim] = local_perm
        phase[site * dim:(site + 1) * dim] = local_phase
    return _edkern.MonomialAction(action.site.perm.astype(int).tolist(),
                                  perm.tolist(), phase.tolist())


def translations(model: 'Model') -> list:
    '''
    The translations, one kernel action each.

    Args:
        model: the model.
    Returns:
        A list of ``MonomialAction``, in the order of ``model.tr_group``.
    '''
    require()
    sites, dim = model.size_of_unit_cell, model.physical_dim
    identity = np.tile(np.arange(dim), sites).tolist()
    return [_edkern.MonomialAction(model._translation_perm(t).perm.astype(int).tolist(),
                                   identity, [1.0 + 0.0j] * (sites * dim))
            for t in model.tr_group]


def hilbert_index(model: 'Model', op) -> int:
    '''
    Index of the point-group action that belongs to an operation matrix.

    Args:
        model: the model.
        op: an operation of ``model.point_group``.
    Returns:
        The index into ``model.hilbert``.
    Raises:
        ValueError: if the operation is not part of the model.
    '''
    arr = np.asarray(op, dtype=float)
    for index, action in enumerate(model.actions):
        if np.allclose(action.rotation, arr):
            return index
    raise ValueError('the operation is not part of the model')


def momentum_sector(model: 'Model', k, sztot: Optional[int] = None):
    '''
    The momentum-adapted columns of one sector, as the kernel has them.

    Args:
        model: the model.
        k: momentum label.
        sztot: total charge, or None for the whole space.
    Returns:
        ``(basis, columns)``: the kernel's ``Basis`` and its ``SparseColumns``.
        The tests use this to reach the momentum basis without going through
        the sector decomposition.
    '''
    require()
    basis = basis_of(model, sztot)
    characters = [complex(model.tr_group.character(k, t)) for t in model.tr_group]
    return basis, _edkern.momentum_basis(basis, translations(model), characters)


# ── sparse results, as scipy matrices ─────────────────────────────────────────

def to_scipy(matrix) -> sp.csr_matrix:
    '''
    A kernel CSR matrix as a scipy matrix.

    Args:
        matrix: ``_edkern.SparseMatrix``.
    Returns:
        The same matrix, in CSR form.
    '''
    return sp.csr_matrix((matrix.data, matrix.indices, matrix.indptr),
                         shape=(matrix.rows, matrix.cols))


def columns_to_scipy(columns, states: int) -> sp.csc_matrix:
    '''
    A set of adapted columns as a scipy matrix.

    Args:
        columns: ``_edkern.SparseColumns``; compressed by column.
        states: number of rows, i.e. size of the computational basis.
    Returns:
        The ``(states, columns.columns())`` matrix.
    '''
    return sp.csc_matrix((columns.data, columns.indices, columns.indptr),
                         shape=(states, columns.columns()))


def vectors_to_scipy(vectors, columns: int) -> sp.csc_matrix:
    '''
    One isotypic basis as a scipy matrix.

    Args:
        vectors: ``_edkern.SectorVectors``; compressed by vector.
        columns: number of rows, i.e. size of the sector.
    Returns:
        The ``(columns, vectors.vectors())`` matrix.
    '''
    return sp.csc_matrix((vectors.data, vectors.columns, vectors.start),
                         shape=(columns, vectors.vectors()))


# ── the sector decomposition ──────────────────────────────────────────────────

#: How many workers the kernel may use inside one sector; None means one per core,
#: and the kernels clamp that to the work they have.
workers: Optional[int] = None


def _threads(threads: Optional[int] = None) -> int:
    '''
    The thread count to hand the kernel.

    Args:
        threads: an explicit count, or None to use the module setting.
    Returns:
        The count, where 0 means one per hardware thread.
    '''
    if threads is not None:
        return int(threads)
    return 0 if workers is None else int(workers)


def _identity_permutation(count: int) -> '_edkern.PhasedPermutation':
    '''The identity as a phased permutation of *count* columns.'''
    return _edkern.PhasedPermutation(list(range(count)), [1.0 + 0.0j] * count)


def _compose(first, second, count: int) -> '_edkern.PhasedPermutation':
    '''
    The phased permutation ``first * second``.

    Args:
        first: the later operation.
        second: the earlier one.
        count: number of columns.
    Returns:
        ``(act, phase)`` with ``act[x] = first.act[second.act[x]]`` and the
        phases multiplied along the way.
    '''
    act = [int(first.act[int(second.act[x])]) for x in range(count)]
    phase = [second.phase[x] * first.phase[int(second.act[x])] for x in range(count)]
    return _edkern.PhasedPermutation(act, phase)


def _little_group(model: 'Model', columns, basis, k: tuple[int, ...], threads: int):
    '''
    The little group of one momentum, as actions on the sector's columns.

    Args:
        model: the model.
        columns: the momentum basis of *k*.
        basis: the computational basis.
        k: the momentum label.
        threads: workers for the actions.
    Returns:
        ``(actions, irreps)``: one action per little-group element -- the identity
        alone when the group is trivial -- and one entry per irrep of it, as
        ``(label, dimension, characters)``.
    '''
    little, _ = model.little_group(k)
    ops = list(little)
    if len(ops) == 1:
        return [_identity_permutation(columns.columns())], [('A', 1, [1.0 + 0.0j])]
    actions = [_edkern.sector_action(
        columns, basis, monomial_of(model, model.hilbert[hilbert_index(model, op)]))
        for op in ops]
    irreps = [(alpha, little.irrep_dim(alpha),
               [little.character(alpha, op) for op in ops]) for alpha in little.all_irrps()]
    return actions, irreps


def _isotypic(actions: list, characters: list, flip, parity, dim: int, count: int,
              threads: int):
    '''
    The adapted vectors of one sector, with the spin flip folded in when asked.

    Args:
        actions: the little-group actions on the sector's columns.
        characters: their characters in the irrep.
        flip: the spin flip acting on the columns, or None.
        parity: the eigenvalue of the spin flip, or None.
        dim: dimension of the irrep.
        count: number of columns of the sector.
        threads: workers for the projection.
    Returns:
        The orthonormal adapted vectors.
    '''
    if parity is None:
        return _edkern.isotypic_basis(actions, characters, dim, count, threads)
    # the spin flip commutes with the whole group, so the resolution is the product
    # irrep of G_k x Z_2: the elements are (r, 0) and (r, 1), and the characters are
    # chi_alpha(r) and s chi_alpha(r)
    both = list(actions) + [_compose(action, flip, count) for action in actions]
    both_characters = list(characters) + [value * parity for value in characters]
    return _edkern.isotypic_basis(both, both_characters, dim, count, threads)


def _momentum_pieces(model: 'Model', basis, terms, translations, spin, spin_flip: bool,
                     threads: int):
    '''
    One momentum at a time: its columns, its Hamiltonian and its sectors.

    Args:
        model: the model.
        basis: the computational basis.
        terms: the couplings in the kernel's representation.
        translations: the translations as kernel actions.
        spin: the spin flip as a kernel action, or None.
        spin_flip: whether the spin flip is resolved.
        threads: workers for the kernels.
    Yields:
        ``(columns, sector, pieces)``, where *pieces* is a list of
        ``((k, alpha, s), vectors)``.
    '''
    for k in model.tr_group.all_momenta():
        characters = [complex(model.tr_group.character(k, t)) for t in model.tr_group]
        columns = _edkern.momentum_basis(basis, translations, characters, threads)
        count = columns.columns()
        if count == 0:
            continue
        sector = _edkern.sector_hamiltonian(basis, terms, columns, threads)
        actions, irreps = _little_group(model, columns, basis, k, threads)
        flip = (_edkern.sector_action(columns, basis, spin) if spin_flip else None)
        pieces = []
        for alpha, dim, chars in irreps:
            for parity in ((None,) if not spin_flip else (1, -1)):
                vectors = _isotypic(actions, chars, flip, parity, dim, count, threads)
                if vectors.vectors() == 0:
                    continue
                pieces.append(((k, alpha, parity), vectors))
        yield columns, sector, pieces


def spectra(model: 'Model', count: int = 5, sztot: Optional[int] = None,
            spin_flip: bool = False, threads: int = 0) -> dict:
    '''
    The lowest levels of every sector, one momentum at a time.

    Nothing but the eigenvalues is kept: the columns, the sector Hamiltonian and the
    block are built for one momentum, solved, and dropped before the next one
    starts.

    Args:
        model: the model.
        count: how many levels per sector.
        sztot: total charge, or None for the whole space.
        spin_flip: if True, resolve the spin-flip parity as well.
        threads: workers for the kernels; 0 means one per core.
    Returns:
        A dict keyed by ``(k, alpha, s)`` of the eigenvalues, ascending.
    '''
    require()
    basis = basis_of(model, sztot)
    terms = coupling_terms(model)
    translations = globals()["translations"](model)
    spin = monomial_of(model, model._spin_flip_action()) if spin_flip else None
    out = {}
    for _, sector, pieces in _momentum_pieces(model, basis, terms, translations, spin,
                                              spin_flip, threads):
        for key, vectors in pieces:
            block = _edkern.block_matrix(sector, vectors, threads)
            asked = min(count, vectors.vectors())
            # a degenerate level needs one vector per copy, and the copies are
            # resolved one cycle at a time, so the budget grows with the request
            out[key] = np.asarray(
                _edkern.lowest_eigenvalues(block, asked, max(200, 40 * asked)))
    return out


def spectra_free(model: 'Model', count: int = 5, sztot: Optional[int] = None,
                 spin_flip: bool = False, threads: int = 0) -> dict:
    '''
    The lowest levels of every sector, without forming a matrix.

    No momentum basis, no sector Hamiltonian and no block is built: the states are
    labelled by their translation orbit once, the little group acts on those labels,
    and the Hamiltonian of a sector is applied to vectors by expanding them onto the
    orbits, applying the terms and projecting back.  The price is that the product
    is paid at every step of the recurrence instead of being read from a matrix,
    which is what a lattice too large for the matrix can afford.

    Args:
        model: the model.
        count: how many levels per sector.
        sztot: total charge, or None for the whole space.
        spin_flip: if True, resolve the spin-flip parity as well.
        threads: workers for the product; 0 means one per core.
    Returns:
        A dict keyed by ``(k, alpha, s)`` of the eigenvalues, ascending.
    '''
    require()
    basis = basis_of(model, sztot)
    terms = coupling_terms(model)
    translations = globals()["translations"](model)
    orbits = _edkern.sector_map(basis, translations)
    spin = monomial_of(model, model._spin_flip_action()) if spin_flip else None
    budget = max(200, 40 * count)
    out = {}
    for k in model.tr_group.all_momenta():
        characters = [complex(model.tr_group.character(k, t)) for t in model.tr_group]
        # the orbits whose column is empty for this momentum are dropped, the same
        # way the momentum basis drops them
        sector = _edkern.sector_subset(orbits, characters)
        if sector.orbits == 0:
            continue
        little, _ = model.little_group(k)
        ops = list(little)
        if len(ops) == 1:
            actions = [_identity_permutation(sector.orbits)]
            irreps = [('A', 1, [1.0 + 0.0j])]
        else:
            actions = [_edkern.action_on_orbits(
                basis, sector, monomial_of(model, model.hilbert[hilbert_index(model, op)]))
                for op in ops]
            irreps = [(alpha, little.irrep_dim(alpha),
                       [little.character(alpha, op) for op in ops])
                      for alpha in little.all_irrps()]
        flip = (_edkern.action_on_orbits(basis, sector, spin) if spin_flip else None)
        for alpha, dim, chars in irreps:
            for parity in ((None,) if not spin_flip else (1, -1)):
                vectors = _isotypic(actions, chars, flip, parity, dim, sector.orbits, threads)
                if vectors.vectors() == 0:
                    continue
                ask = min(count, vectors.vectors())
                out[(k, alpha, parity)] = np.asarray(_edkern.lowest_eigenvalues_free(
                    lambda vector: _edkern.sector_apply_block(
                        basis, terms, sector, characters, vectors, vector, threads),
                    vectors.vectors(), ask, max(budget, 40 * ask), 1e-10, threads))
    return out


def sectors(model: 'Model', sztot: Optional[int] = None,
            spin_flip: bool = False) -> dict:
    '''
    The whole decomposition, one momentum at a time.

    The same walk as :func:`spectra`, except that the adapted vectors are kept
    beside their block and the block is not diagonalized.

    Args:
        model: the model.
        sztot: total charge, or None for the whole space.
        spin_flip: if True, resolve the parity of the spin flip as well.
    Returns:
        A dict keyed by ``(k, alpha, s)`` of ``(basis, block)``, both CSR: the
        adapted vectors as columns in the computational basis, and the Hamiltonian
        between them.
    '''
    require()
    basis = basis_of(model, sztot)
    terms = coupling_terms(model)
    translations = globals()["translations"](model)
    spin = monomial_of(model, model._spin_flip_action()) if spin_flip else None
    threads = _threads()
    out = {}
    for columns, sector, pieces in _momentum_pieces(model, basis, terms, translations,
                                                    spin, spin_flip, threads):
        for key, vectors in pieces:
            block = to_scipy(_edkern.block_matrix(sector, vectors, threads))
            adapted = (columns_to_scipy(columns, basis.size())
                       @ vectors_to_scipy(vectors, columns.columns()))
            out[key] = (adapted.tocsr(), block.tocsr())
    return out
