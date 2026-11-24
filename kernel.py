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


def _little_group(model: 'Model', sector, basis, k, characters, translations):
    '''
    The little group of one momentum, acting on the orbit labels.

    Args:
        model: the model.
        sector: the orbit labels of the momentum sector.
        basis: the computational basis.
        k: the momentum label.
        characters: the character of the momentum at every translation.
        translations: the translations as kernel actions.
    Returns:
        ``(actions, irreps)``: one action per little-group element -- the identity
        alone when the group is trivial -- and one entry per irrep of it, as
        ``(label, dimension, characters)``.
    '''
    little, _ = model.little_group(k)
    ops = list(little)
    if len(ops) == 1:
        return ([_identity_permutation(sector.orbits)], [('A', 1, [1.0 + 0.0j])])
    actions = [_edkern.action_on_orbits(
        basis, sector, monomial_of(model, model.hilbert[hilbert_index(model, op)]),
        characters, translations) for op in ops]
    irreps = [(alpha, little.irrep_dim(alpha),
               [little.character(alpha, op) for op in ops]) for alpha in little.all_irrps()]
    return actions, irreps


def _isotypic(actions: list, characters: list, flip, parity, dim: int, count: int,
              threads: int):
    '''
    The adapted vectors of one sector, with the spin flip folded in when asked.

    Args:
        actions: the little-group actions on the orbit labels.
        characters: their characters in the irrep.
        flip: the spin flip acting on the labels, or None.
        parity: the eigenvalue of the spin flip, or None.
        dim: dimension of the irrep.
        count: number of orbits in the sector.
        threads: workers for the projection.
    Returns:
        The orthonormal adapted vectors, one entry per orbit.
    '''
    if parity is None:
        return _edkern.isotypic_basis(actions, characters, dim, count, threads)
    # the spin flip commutes with the whole group, so the resolution is the product
    # irrep of G_k x Z_2: the elements are (r, 0) and (r, 1), and the characters are
    # chi_alpha(r) and s chi_alpha(r)
    both = list(actions) + [_compose(action, flip, count) for action in actions]
    both_characters = list(characters) + [value * parity for value in characters]
    return _edkern.isotypic_basis(both, both_characters, dim, count, threads)


def _sector_pieces(model: 'Model', basis, orbits, spin, spin_flip: bool, threads: int):
    '''
    One sector at a time: its labels, its character and its adapted vectors.

    Args:
        model: the model.
        basis: the computational basis.
        orbits: the orbit labels of the whole basis, from ``sector_map``.
        spin: the spin flip as a kernel action, or None.
        spin_flip: whether the spin flip is resolved.
        threads: workers for the projection.
    Yields:
        ``((k, alpha, s), sector, characters, vectors)``.
    '''
    translations = globals()["translations"](model)
    for k in model.tr_group.all_momenta():
        characters = [complex(model.tr_group.character(k, t)) for t in model.tr_group]
        # the orbits whose column is empty for this momentum are dropped, the same
        # way the momentum basis drops them
        sector = _edkern.sector_subset(orbits, characters)
        if sector.orbits == 0:
            continue
        actions, irreps = _little_group(model, sector, basis, k, characters, translations)
        flip = (_edkern.action_on_orbits(basis, sector, spin, characters, translations)
                if spin_flip else None)
        for alpha, dim, chars in irreps:
            for parity in ((None,) if not spin_flip else (1, -1)):
                vectors = _isotypic(actions, chars, flip, parity, dim, sector.orbits, threads)
                if vectors.vectors() == 0:
                    continue
                yield (k, alpha, parity), sector, characters, vectors


def spectra(model: 'Model', count: int = 5, sztot: Optional[int] = None,
            spin_flip: bool = False, threads: int = 0) -> dict:
    '''
    The lowest levels of every sector.

    Each sector's block is built directly from its character -- no momentum basis
    and no momentum-space Hamiltonian -- and diagonalized while the next sector
    waits, so the memory does not grow with the number of sectors.

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
    orbits = _edkern.sector_map(basis, globals()["translations"](model))
    spin = monomial_of(model, model._spin_flip_action()) if spin_flip else None
    out = {}
    for key, sector, characters, vectors in _sector_pieces(model, basis, orbits, spin,
                                                           spin_flip, threads):
        block = _edkern.sector_block(basis, terms, sector, characters, vectors, threads)
        asked = min(count, vectors.vectors())
        # a degenerate level needs one vector per copy, and the copies are resolved
        # one cycle at a time, so the budget grows with the request
        out[key] = np.asarray(_edkern.lowest_eigenvalues(block, asked, max(200, 40 * asked)))
    return out


def spectra_free(model: 'Model', count: int = 5, sztot: Optional[int] = None,
                 spin_flip: bool = False, threads: int = 0) -> dict:
    '''
    The lowest levels of every sector, without forming a block.

    The same walk as :func:`spectra`, but the product is applied to vectors instead
    of being written down: what remains in memory is the vector alone.

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
    orbits = _edkern.sector_map(basis, globals()["translations"](model))
    spin = monomial_of(model, model._spin_flip_action()) if spin_flip else None
    out = {}
    for key, sector, characters, vectors in _sector_pieces(model, basis, orbits, spin,
                                                           spin_flip, threads):
        asked = min(count, vectors.vectors())
        out[key] = np.asarray(_edkern.lowest_eigenvalues_free(
            lambda vector: _edkern.sector_apply_block(basis, terms, sector, characters,
                                                      vectors, vector, threads),
            vectors.vectors(), asked, max(200, 40 * asked), 1e-10, threads))
    return out


def sectors(model: 'Model', sztot: Optional[int] = None, spin_flip: bool = False) -> dict:
    '''
    The whole decomposition, one sector at a time.

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
    orbits = _edkern.sector_map(basis, globals()["translations"](model))
    spin = monomial_of(model, model._spin_flip_action()) if spin_flip else None
    threads = _threads()
    out = {}
    for key, sector, characters, vectors in _sector_pieces(model, basis, orbits, spin,
                                                           spin_flip, threads):
        block = to_scipy(_edkern.sector_block(basis, terms, sector, characters, vectors, threads))
        adapted = to_scipy(_edkern.sector_basis(basis, sector, characters, vectors, threads))
        out[key] = (adapted.tocsr(), block.tocsr())
    return out
