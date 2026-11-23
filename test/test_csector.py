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
from scipy.sparse import csr_matrix

twod = importlib.import_module('geometry.2dsym')
from ed import Coupling, Model
from hilbert_action import circle_doublet

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

def test_hamiltonian(model: Model, Sztot, label: str, verbose: bool = True) -> bool:
    '''The assembled Hamiltonian equals the Python one, and respects the group.'''
    basis = cxx_basis(model, Sztot)
    couplings = cxx_couplings(model)
    actions = cxx_group_actions(model)
    raw = _edkern.assemble(basis, couplings)
    kernel = _edkern.symmetrise(raw, basis, actions)
    reference = model._build_H_sym(Sztot)
    difference = abs(to_scipy(kernel) - reference).max() if kernel.nonzeros() else 0.0
    # the couplings are supposed to respect the group, so averaging must be a
    # no-op; if it is not, the fixture is not the model its label claims
    drift = abs(to_scipy(kernel) - to_scipy(raw)).max() if raw.nonzeros() else 0.0
    ok = difference < 1e-12 and drift < 1e-12
    if verbose:
        print(f"  {'✓' if ok else 'FAIL'}  {label} H: {kernel.nonzeros()} non-zeros, "
              f"max difference {difference:.1e}, symmetrisation drift {drift:.1e}")
    return ok


def sectors_of(model: Model, Sztot):
    '''The Python sector decomposition, keyed by (k, irrep).'''
    return {key: sector for key, sector in model.build_sectors(Sztot=Sztot).items()}


def test_sector_blocks(model: Model, Sztot, label: str, tol: float = 1e-9,
                       verbose: bool = True) -> bool:
    '''Sector dimensions and block spectra equal the Python ones.'''
    basis = cxx_basis(model, Sztot)
    couplings = cxx_couplings(model)
    actions = cxx_group_actions(model)
    hamiltonian = _edkern.symmetrise(_edkern.assemble(basis, couplings), basis, actions)
    reference = sectors_of(model, Sztot)
    translations = [cxx_translation(model, t) for t in model.tr_group]

    worst_dim = 0
    worst_energy = 0.0
    worst_block = 0.0
    checked = 0
    for k in model.tr_group.all_momenta():
        characters = [complex(model.tr_group.character(k, t)) for t in model.tr_group]
        columns = _edkern.momentum_basis(basis, translations, characters)
        if columns.columns() == 0:
            continue
        sector_hamiltonian = _edkern.sector_hamiltonian(hamiltonian, columns)
        little_group, _ = model.little_group(k)
        ops = list(little_group)
        if len(ops) == 1:
            continue
        permutations = [_edkern.sector_action(columns, basis, cxx_action(model, op))
                        for op in ops]
        for alpha in little_group.all_irrps():
            characters_alpha = [little_group.character(alpha, op) for op in ops]
            vectors = _edkern.isotypic_basis(permutations, characters_alpha,
                                             little_group.irrep_dim(alpha),
                                             columns.columns())
            if vectors.vectors() == 0:
                continue
            # the block comes back sparse; the dense matrix it stands for is
            # built here, in the test, and diagonalised with numpy
            block = to_scipy(_edkern.block_matrix(sector_hamiltonian, vectors))
            energies = np.sort(np.linalg.eigvalsh(block.toarray()))
            reference_sector = reference.get((k, alpha, None))
            want = (np.sort(np.linalg.eigvalsh(reference_sector.hamiltonian.toarray()))
                    if reference_sector is not None else np.array([]))
            worst_dim = max(worst_dim, abs(len(energies) - len(want)))
            # the sparse block against V^T H_k V formed densely from the same
            # vectors: an oracle for block_matrix that does not reuse its code
            dense_vectors = np.zeros((columns.columns(), vectors.vectors()), dtype=complex)
            for c in range(vectors.vectors()):
                for a in range(vectors.start[c], vectors.start[c + 1]):
                    dense_vectors[vectors.columns[a], c] = vectors.data[a]
            dense_block = (dense_vectors.conj().T
                           @ (to_scipy(sector_hamiltonian).toarray() @ dense_vectors))
            worst_block = max(worst_block,
                              float(np.max(np.abs(block.toarray() - dense_block))))
            if len(want):
                worst_energy = max(worst_energy, float(np.max(np.abs(energies - want))))
            checked += len(energies)
    ok = worst_dim == 0 and worst_energy < tol and worst_block < tol
    if verbose:
        print(f"  {'✓' if ok else 'FAIL'}  {label} sectors: {checked} eigenvalues over all "
              f"(k, alpha), dimension mismatch {worst_dim}, max |dE| {worst_energy:.1e}, "
              f"|block - V^T H_k V| {worst_block:.1e}")
    return ok


def test_lanczos(model: Model, Sztot, label: str, tol: float = 1e-8,
                 verbose: bool = True) -> bool:
    '''Lanczos reproduces the lowest dense eigenvalues of a sector Hamiltonian.'''
    basis = cxx_basis(model, Sztot)
    couplings = cxx_couplings(model)
    actions = cxx_group_actions(model)
    hamiltonian = _edkern.symmetrise(_edkern.assemble(basis, couplings), basis, actions)
    translations = [cxx_translation(model, t) for t in model.tr_group]
    k = model.tr_group.all_momenta()[0]
    characters = [complex(model.tr_group.character(k, t)) for t in model.tr_group]
    columns = _edkern.momentum_basis(basis, translations, characters)
    sector_hamiltonian = _edkern.sector_hamiltonian(hamiltonian, columns)
    dense = np.sort(np.linalg.eigvalsh(to_scipy(sector_hamiltonian).toarray()))
    # a single-vector recursion resolves distinct levels only, so compare
    # against the distinct part of the dense spectrum
    distinct = np.unique(np.round(dense, 10))
    count = min(4, len(distinct))
    found = np.sort(np.asarray(_edkern.lowest_eigenvalues(sector_hamiltonian, count)))
    difference = float(np.max(np.abs(found - distinct[:count]))) if count else 0.0
    # a recurrence cut off after two steps cannot have converged: it has to
    # refuse rather than hand back a number between two levels
    refused = False
    try:
        _edkern.lowest_eigenvalues(sector_hamiltonian, count, 2)
    except RuntimeError:
        refused = True
    ok = difference < tol and refused
    if verbose:
        print(f"  {'✓' if ok else 'FAIL'}  {label} Lanczos: {count} lowest distinct levels "
              f"of a {sector_hamiltonian.rows}-dimensional sector, max difference "
              f"{difference:.1e}, unconverged run refused {refused}")
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
        test_hamiltonian(model, sztot, label)
        test_sector_blocks(model, sztot, label)
        test_lanczos(model, sztot, label)
        print()


if __name__ == '__main__':
    main()
