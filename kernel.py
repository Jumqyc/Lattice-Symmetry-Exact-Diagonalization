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

from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy.sparse as sp

try:
    import _edkern
except ImportError:                     # pragma: no cover - the kernel is optional
    _edkern = None

if TYPE_CHECKING:                        # pragma: no cover
    from ed import Model


def available() -> bool:
    '''
    Whether the compiled kernel can be imported.

    Returns:
        True if ``_edkern`` is importable.
    '''
    return _edkern is not None


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


def group_actions(model: 'Model') -> list:
    '''
    The whole space group, one kernel action per element.

    Args:
        model: the model.
    Returns:
        Every composition of a translation with a point-group action.  The
        symmetrisation averages over all of them, so the kernel needs the
        composed elements and not the generators.
    '''
    require()
    out = []
    for t in model.tr_group:
        translation = model._translation_perm(t)
        for index in range(len(model.hilbert)):
            out.append(monomial_of(model, model.hilbert[index].after_translation(translation)))
    return out


def _hilbert_for(model: 'Model', op) -> int:
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


def sectors(model: 'Model', sztot: Optional[int] = None,
            spin_flip: bool = False) -> dict:
    '''
    The whole decomposition, built by the kernel.

    Args:
        model: the model.
        sztot: total charge, or None for the whole space.
        spin_flip: if True, resolve the parity of the spin flip as well.
    Returns:
        A dict keyed by ``(k, alpha, s)`` whose values are
        ``(basis, block)``, both CSR matrices.  ``basis`` has the adapted
        vectors as columns, in the computational basis; ``block`` is the
        Hamiltonian between them.
    '''
    require()
    states = model.size_of_unit_cell
    basis = basis_of(model, sztot)
    hamiltonian = _edkern.symmetrise(_edkern.assemble(basis, coupling_terms(model)),
                                     basis, group_actions(model))
    translations = globals()["translations"](model)
    spin = monomial_of(model, model._spin_flip_action()) if spin_flip else None
    out = {}
    for k in model.tr_group.all_momenta():
        characters = [complex(model.tr_group.character(k, t)) for t in model.tr_group]
        columns = _edkern.momentum_basis(basis, translations, characters)
        count = columns.columns()
        if count == 0:
            continue
        sector = _edkern.sector_hamiltonian(hamiltonian, columns)
        little, _ = model.little_group(k)
        ops = list(little)
        identity = _identity_permutation(count)
        if len(ops) == 1:
            actions, irreps = [identity], [('A', 1, [1.0 + 0.0j])]
        else:
            actions = [_edkern.sector_action(columns, basis,
                                             monomial_of(model, model.hilbert[_hilbert_for(model, op)]))
                       for op in ops]
            irreps = [(alpha, little.irrep_dim(alpha),
                       [little.character(alpha, op) for op in ops])
                      for alpha in little.all_irrps()]
        flip = (_edkern.sector_action(columns, basis, spin) if spin_flip else None)
        # the spin flip commutes with the whole group, so the resolution is the
        # product irrep of G_k x Z_2: the elements are (r, 0) and (r, 1), the
        # characters are chi_alpha(r) and s chi_alpha(r), and the projector is
        # the one the kernel already knows how to build
        for alpha, dim, chars in irreps:
            for s in ((None,) if not spin_flip else (1, -1)):
                if s is None:
                    use_actions, use_chars = actions, chars
                else:
                    use_actions = list(actions) + [_compose(action, flip, count)
                                                   for action in actions]
                    use_chars = list(chars) + [value * s for value in chars]
                vectors = _edkern.isotypic_basis(use_actions, use_chars, dim, count)
                if vectors.vectors() == 0:
                    continue
                block = to_scipy(_edkern.block_matrix(sector, vectors))
                adapted = columns_to_scipy(columns, basis.size()) @ vectors_to_scipy(vectors, count)
                out[(k, alpha, s)] = (adapted.tocsr(), block.tocsr())
    return out
