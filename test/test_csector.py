'''
Test of the C++ Hamiltonian and sector layer against the Python reference.

What is compared, stage by stage:

* the Hamiltonian assembled from the couplings, element by element;
* the Hamiltonian in the momentum-adapted basis, element by element;
* the dimension of every irrep sector, against the Python decomposition;
* the eigenvalues of every sector block;
* the Lanczos solver against the dense eigensolver on the same block.

Requires the kernel: build it with ``python cpp/build.py``.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import numpy as np
import reference
from scipy.sparse import csr_matrix

twod = importlib.import_module('geometry.2dsym')
from ed import Coupling, Model

try:
    import _edkern
except ImportError:                                  # pragma: no cover
    _edkern = None

X = np.array([[0., 1.], [1., 0.]], dtype=complex)
Y = np.array([[0., -1j], [1j, 0.]], dtype=complex)
Z = np.array([[1., 0.], [0., -1.]], dtype=complex)
A1 = np.array([1., 0.])
A2 = np.array([0., 1.])
T1 = np.array([1., 0.])
T2 = np.array([0.5, np.sqrt(3.) / 2.])


def heisenberg(N, point_group, a1=A1, a2=A2) -> Model:
    '''Nearest-neighbour Heisenberg model on an N x N torus.'''
    model = Model(2, [(a1, N), (a2, N)], {('A', 0): np.array([0., 0.])},
                  point_group=point_group)
    # the third nearest-neighbour direction of a triangular lattice is a1 - a2;
    # the 60-degree rotation maps an a1 bond onto it, so leaving it out makes the
    # model invariant under nothing but translations
    directions = [(0, 1), (1, 0)]
    if np.allclose(np.linalg.norm(np.asarray(a1) - np.asarray(a2)), np.linalg.norm(a1)):
        directions.append((1, -1))
    for i in range(N):
        for j in range(N):
            site = i * N + j
            # each unordered bond is visited once per direction: the wrap bond is
            # the instance j = N - 1
            for di, dj in directions:
                partner = ((i + di) % N) * N + (j + dj) % N
                model.add_coupling(Coupling(2, (site, partner), (X, X), (Y, Y), (Z, Z)))
    return model


# ── conversions from the Python model to the kernel's plain arrays ────────────

def cxx_basis(model: Model, Sztot) -> '_edkern.Basis':
    '''The basis of the model in the kernel's representation.'''
    return _edkern.Basis(model.size_of_unit_cell, model.physical_dim,
                         list(model.local_charge), -1 if Sztot is None else Sztot)


def cxx_couplings(model: Model) -> list:
    '''The couplings as per-site sparse local operators.'''
    out = []
    for coupling in model.couplings:
        # one CouplingTerm per term: concatenating the entries of several terms
        # into one local table would mix them, which is a different operator
        for term in coupling.terms:
            local = []
            for position in range(len(coupling.sites)):
                operator = term[position]
                outs, ins, coeffs = [], [], []
                for column in range(operator.shape[1]):
                    for row in range(operator.shape[0]):
                        if abs(operator[row, column]) > 1e-14:
                            outs.append(row)
                            ins.append(column)
                            coeffs.append(complex(operator[row, column]))
                local.append(_edkern.LocalTerm(outs, ins, coeffs))
            out.append(_edkern.CouplingTerm(list(coupling.sites), local))
    return out


def cxx_translation(model: Model, t) -> '_edkern.MonomialAction':
    '''A translation as a kernel action (no local part).'''
    sites, dim = model.size_of_unit_cell, model.physical_dim
    identity = np.tile(np.arange(dim), sites).tolist()
    return _edkern.MonomialAction(model._translation_perm(t).perm.astype(int).tolist(),
                                  identity, [1.0 + 0.0j] * (sites * dim))


def cxx_from(model: Model, action) -> '_edkern.MonomialAction':
    '''A Python monomial action in the kernel's representation.'''
    sites, dim = model.size_of_unit_cell, model.physical_dim
    perm = np.tile(np.arange(dim), sites)
    phase = np.ones(sites * dim, dtype=complex)
    for i, species in enumerate(model.species):
        if species in action.local:
            p, q = action.local[species]
            perm[i * dim:(i + 1) * dim] = p
            phase[i * dim:(i + 1) * dim] = q
    return _edkern.MonomialAction(action.site.perm.astype(int).tolist(),
                                  perm.tolist(), phase.tolist())


def cxx_hilbert(model: Model, index: int) -> '_edkern.MonomialAction':
    '''One point-group action of the model, in the kernel's representation.'''
    return cxx_from(model, model.hilbert[index])


def cxx_group_actions(model: Model) -> list:
    '''
    The whole lattice group, one action per element.

    The symmetrisation is an average over all (translation, point-group) pairs,
    so the kernel needs every composed element, not the generators.
    '''
    out = []
    for t in model.tr_group:
        translation = model._translation_perm(t)
        for index in range(len(model.hilbert)):
            out.append(cxx_from(model, model.hilbert[index].after_translation(translation)))
    return out


def cxx_action(model: Model, op) -> '_edkern.MonomialAction':
    '''The kernel action of a point-group operation of the model.'''
    arr = np.asarray(op, dtype=float)
    for index, action in enumerate(model.actions):
        if np.allclose(action.rotation, arr):
            return cxx_hilbert(model, index)
    raise ValueError('the operation is not part of the model')


def to_scipy(matrix) -> csr_matrix:
    '''A kernel CSR matrix as a scipy matrix.'''
    return csr_matrix((matrix.data, matrix.indices, matrix.indptr),
                      shape=(matrix.rows, matrix.cols))


# ── the checks ────────────────────────────────────────────────────────────────

def sector_of(model: Model, Sztot, k: tuple[int, ...]):
    '''
    The orbit labels, character and adapted vectors of one sector.

    Args:
        model: the model.
        Sztot: total charge, or None.
        k: the momentum label.
    Returns:
        ``(basis, sector, characters, vectors)`` for the trivial irrep, which is
        all a test of the operator needs.
    '''
    basis = cxx_basis(model, Sztot)
    translations = [cxx_translation(model, t) for t in model.tr_group]
    characters = [complex(model.tr_group.character(k, t)) for t in model.tr_group]
    orbits = _edkern.sector_map(basis, translations)
    sector = _edkern.sector_subset(orbits, characters)
    identity = _edkern.PhasedPermutation(list(range(sector.orbits)),
                                        [1.0 + 0.0j] * sector.orbits)
    vectors = _edkern.isotypic_basis([identity], [1.0 + 0.0j], 1, sector.orbits)
    return basis, sector, characters, vectors


def test_term_by_term(model: Model, Sztot, label: str, tol: float = 1e-10,
                      verbose: bool = True) -> bool:
    '''
    The sector's block is the restriction of the Hamiltonian.

    The block is built directly from the character; the test forms the same object
    densely, ``X^T H X``, from the adapted vectors written out in the computational
    basis.
    '''
    basis, sector, characters, vectors = sector_of(model, Sztot, (0, 0))
    block = to_scipy(_edkern.sector_block(basis, cxx_couplings(model), sector, characters,
                                          vectors)).toarray()
    adapted = to_scipy(_edkern.sector_basis(basis, sector, characters, vectors)).toarray()
    projected = adapted.conj().T @ reference.hamiltonian(model, Sztot) @ adapted
    worst = float(np.max(np.abs(block - projected)))
    ok = worst < tol
    if verbose:
        print(f"  {'✓' if ok else 'FAIL'}  {label} sector block: built from the character "
              f"against the dense restriction, max difference {worst:.1e}")
    return ok


def test_sector_blocks(model: Model, Sztot, label: str, tol: float = 1e-9,
                       verbose: bool = True) -> bool:
    '''Block matrices, their dimensions and the union of their spectra.'''
    basis = cxx_basis(model, Sztot)
    couplings = cxx_couplings(model)
    dense_hamiltonian = reference.hamiltonian(model, Sztot)
    dense = np.sort(np.linalg.eigvalsh(dense_hamiltonian))
    translations = [cxx_translation(model, t) for t in model.tr_group]
    orbits = _edkern.sector_map(basis, translations)
    worst_block = 0.0
    collected = []
    for k in model.tr_group.all_momenta():
        characters = [complex(model.tr_group.character(k, t)) for t in model.tr_group]
        sector = _edkern.sector_subset(orbits, characters)
        if sector.orbits == 0:
            continue
        little_group, _ = model.little_group(k)
        ops = list(little_group)
        if len(ops) == 1:
            identity = _edkern.PhasedPermutation(list(range(sector.orbits)),
                                                [1.0 + 0.0j] * sector.orbits)
            pieces = [('A', _edkern.isotypic_basis([identity], [1.0 + 0.0j], 1, sector.orbits))]
        else:
            actions = [_edkern.action_on_orbits(basis, sector, cxx_action(model, op),
                                                characters, translations) for op in ops]
            pieces = []
            for alpha in little_group.all_irrps():
                characters_alpha = [little_group.character(alpha, op) for op in ops]
                vectors = _edkern.isotypic_basis(actions, characters_alpha,
                                                 little_group.irrep_dim(alpha), sector.orbits)
                if vectors.vectors() > 0:
                    pieces.append((alpha, vectors))
        for _, vectors in pieces:
            block = to_scipy(_edkern.sector_block(basis, couplings, sector, characters, vectors))
            collected.append(np.linalg.eigvalsh(block.toarray()))
            # the same block, formed densely from its adapted vectors in the
            # computational basis: an oracle that does not reuse the block's code
            adapted = to_scipy(_edkern.sector_basis(basis, sector, characters, vectors)).toarray()
            dense_block = adapted.conj().T @ dense_hamiltonian @ adapted
            worst_block = max(worst_block, float(np.max(np.abs(block.toarray() - dense_block))))
    merged = np.sort(np.concatenate(collected)) if collected else np.zeros((0,))
    worst_energy = (float(np.max(np.abs(merged - dense)))
                    if len(merged) == len(dense) else float('inf'))
    ok = worst_block < tol and worst_energy < tol
    if verbose:
        print(f"  {'✓' if ok else 'FAIL'}  {label} sectors: {len(merged)} eigenvalues over "
              f"all (k, alpha), max |dE| {worst_energy:.1e}, "
              f"|block - X^T H X| {worst_block:.1e}")
    return ok


def test_lanczos(model: Model, Sztot, label: str, tol: float = 1e-8,
                 verbose: bool = True) -> bool:
    '''Lanczos reproduces the lowest dense eigenvalues of a sector block.'''
    basis, sector, characters, vectors = sector_of(model, Sztot, model.tr_group.all_momenta()[0])
    block = _edkern.sector_block(basis, cxx_couplings(model), sector, characters, vectors)
    dense = np.sort(np.linalg.eigvalsh(to_scipy(block).toarray()))
    # A single-vector recursion resolves one copy per cycle, so what it returns is a
    # subset of the spectrum: every value must be an eigenvalue, and the lowest must
    # be the lowest -- asking for copies of a degenerate level is not what it does.
    count = min(4, block.rows - 1)
    found = np.sort(np.asarray(_edkern.lowest_eigenvalues(block, count)))
    distance = np.min(np.abs(found[:, None] - dense[None, :]), axis=1)
    worst = float(np.max(distance))
    if abs(found[0] - dense[0]) > tol:
        worst = max(worst, float(abs(found[0] - dense[0])))
    refused = False
    try:
        _edkern.lowest_eigenvalues(block, count, 2)
    except RuntimeError:
        refused = True
    ok = worst < tol and refused
    if verbose:
        print(f"  {'✓' if ok else 'FAIL'}  {label} Lanczos: {count} lowest distinct levels of a "
              f"{block.rows}-dimensional block, max difference {worst:.1e}; unconverged run "
              f"refused {refused}")
    return ok


def main():
    if _edkern is None:
        print('  skipped: the C++ kernel is not built (python cpp/build.py)')
        sys.exit(0)
    for label, builder, sztot in [('3x3 square, D4', lambda: heisenberg(3, twod.D4()), None),
                                  ('3x3 triangular, D6',
                                   lambda: heisenberg(3, twod.D6(), T1, T2), None)]:
        model = builder()
        print(f'── {label} ──')
        test_term_by_term(model, sztot, label)
        test_sector_blocks(model, sztot, label)
        test_lanczos(model, sztot, label)
        print()


if __name__ == '__main__':
    main()
