'''
Test of the state layer: the basis, the action of the group, and the momentum
basis.

Each of the three is checked against something it cannot influence:

* the basis against the tuples ``itertools`` writes down, which is the
  definition of the computational basis;
* the action of every group element against the action layer's ``on_state``,
  which is where the site permutation and the local phase are defined;
* the momentum basis against the properties it has to have -- orthonormal
  columns, an eigenvalue of every translation, and as many columns as the charge
  sector has states.

Requires the kernel: build it with ``python cpp/build.py``.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import numpy as np

import reference

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


def heisenberg(N, point_group, a1=A1, a2=A2, orbital_mixing=None, local_charge=None) -> Model:
    '''Nearest-neighbour Heisenberg model on an N x N torus.'''
    dim = 2 if orbital_mixing is None else 4
    model = Model(dim, [(a1, N), (a2, N)], {('A', 0): np.array([0., 0.])},
                  point_group=point_group, orbital_mixing=orbital_mixing,
                  local_charge=local_charge)
    # the Heisenberg bond is the identity on the orbital factor, so that the
    # model commutes with the site permutation times the orbital rotation
    local_ops = ((X, Y, Z) if orbital_mixing is None
                 else tuple(np.kron(np.eye(2, dtype=complex), M) for M in (X, Y, Z)))
    # a1 - a2 closes the triangular lattice's third neighbour direction
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
                model.add_coupling(Coupling(dim, (site, partner),
                                            (local_ops[0], local_ops[0]),
                                            (local_ops[1], local_ops[1]),
                                            (local_ops[2], local_ops[2])))
    return model


def local_tables(model: Model, action) -> tuple[list[int], list[complex]]:
    '''
    Expand a Python monomial action into per-site local tables.

    Args:
        model: the model, for the site count and the species of every site.
        action: a ``hilbert_action.MonomialAction``.
    Returns:
        (local_perm, local_phase), flattened as ``site * local_dim + value``.
    '''
    sites, dim = model.size_of_unit_cell, model.physical_dim
    perm = np.tile(np.arange(dim), sites)
    phase = np.ones(sites * dim, dtype=complex)
    for i, species in enumerate(model.species):
        if species in action.local:
            p, q = action.local[species]
            perm[i * dim:(i + 1) * dim] = p
            phase[i * dim:(i + 1) * dim] = q
    return perm.tolist(), phase.tolist()


def translation_action(model: Model, t) -> '_edkern.MonomialAction':
    '''The C++ action of one translation.'''
    perm, phase = local_tables(model, model.hilbert[0])   # identity local action
    return _edkern.MonomialAction(model._translation_perm(t).perm.astype(int).tolist(),
                                  perm, [1.0 + 0.0j] * len(phase))


def cxx_action(model: Model, op) -> '_edkern.MonomialAction':
    '''The C++ action of one point-group operation.'''
    action = model._action_for(op)
    perm, phase = local_tables(model, action)
    return _edkern.MonomialAction(action.site.perm.astype(int).tolist(), perm, phase)


def test_basis(model: Model, Sztot, label: str, verbose: bool = True) -> bool:
    '''The enumerated basis is the one itertools writes down.'''
    states = reference.basis_states(model, Sztot)
    basis = _edkern.Basis(model.size_of_unit_cell, model.physical_dim,
                          list(model.local_charge), -1 if Sztot is None else Sztot)
    same_size = basis.size() == len(states)
    same_states = same_size and all(
        tuple(basis.unpack(basis.state(i))) == states[i] for i in range(basis.size()))
    same_index = same_states and all(basis.index(basis.state(i)) == i
                                     for i in range(basis.size()))
    ok = same_size and same_states and same_index
    if verbose:
        print(f"  {'✓' if ok else 'FAIL'}  {label}: |H| = {basis.size()}/{len(states)}, "
              f"same order {same_states}, index map {same_index}")
    return ok


def test_action(model: Model, Sztot, label: str, verbose: bool = True) -> bool:
    '''Every group element acts as in Python, phase included.'''
    states = reference.basis_states(model, Sztot)
    basis = _edkern.Basis(model.size_of_unit_cell, model.physical_dim,
                          list(model.local_charge), -1 if Sztot is None else Sztot)
    worst = 0.0
    ok = True
    index_of = {state: row for row, state in enumerate(states)}
    probe = states[:: max(1, len(states) // 16)]          # a spread of states
    for index, action in enumerate(model.hilbert):
        cxx = cxx_action(model, model.actions[index].rotation)
        for state in probe:
            image, phase = _edkern.apply(basis, cxx, basis.state(index_of[state]))
            py_image, py_phase = action.on_state(state)
            ok &= tuple(basis.unpack(image)) == py_image
            worst = max(worst, abs(complex(phase) - py_phase))
    ok &= worst < 1e-12
    if verbose:
        print(f"  {'✓' if ok else 'FAIL'}  {label}: {len(model.hilbert)} elements x "
              f"{len(probe)} states, max phase difference {worst:.1e}")
    return ok


def test_momentum(model: Model, Sztot, label: str, tol: float = 1e-10,
                  verbose: bool = True) -> bool:
    '''
    The momentum basis is orthonormal, translation-diagonal and complete.

    Every column is checked for its norm, a sample of four columns per momentum
    for the eigenvalue property, and the count over all momenta for
    completeness: those three are what a momentum-adapted basis has to satisfy,
    and none of them can be satisfied by accident.
    '''
    basis = _edkern.Basis(model.size_of_unit_cell, model.physical_dim,
                          list(model.local_charge), -1 if Sztot is None else Sztot)
    translations = [translation_action(model, t) for t in model.tr_group]
    norm_error = 0.0
    diagonal_error = 0.0
    columns_total = 0
    for k in model.tr_group.all_momenta():
        characters = [complex(model.tr_group.character(k, t)) for t in model.tr_group]
        columns = _edkern.momentum_basis(basis, translations, characters)
        columns_total += columns.columns()
        stride = max(1, columns.columns() // 4)
        for c in range(columns.columns()):
            support = list(range(columns.indptr[c], columns.indptr[c + 1]))
            norm = sum(abs(complex(columns.data[a])) ** 2 for a in support)
            norm_error = max(norm_error, abs(norm - 1.0))
            if c % stride:
                continue
            column = {int(columns.indices[a]): complex(columns.data[a]) for a in support}
            for t, action in zip(model.tr_group, translations):
                character = complex(model.tr_group.character(k, t))
                image = {}
                for state, amplitude in column.items():
                    moved, phase = _edkern.apply(basis, action, basis.state(state))
                    # apply returns the packed state; the column is keyed by index
                    target = basis.index(moved)
                    image[target] = image.get(target, 0.0) + amplitude * phase
                for state in set(image) | set(column):
                    diagonal_error = max(diagonal_error, abs(image.get(state, 0.0)
                                                             - character * column.get(state, 0.0)))
    complete = columns_total == basis.size()
    ok = norm_error < tol and diagonal_error < tol and complete
    if verbose:
        print(f"  {'✓' if ok else 'FAIL'}  {label}: {columns_total} columns over all momenta "
              f"({basis.size()} states), |norm - 1| {norm_error:.1e}, "
              f"|U(t)v - chi v| {diagonal_error:.1e}, complete {complete}")
    return ok


def main():
    if _edkern is None:
        print('  skipped: the C++ kernel is not built (python cpp/build.py)')
        sys.exit(0)

    orbital = {'A': lambda rot: np.kron(circle_doublet(1)(rot), np.eye(2, dtype=complex))}
    cases = [
        ('3x3 square, D4', lambda: heisenberg(3, twod.D4()), None),
        ('4x4 square, D4, Sz=0', lambda: heisenberg(4, twod.D4()), 8),
        ('3x3 triangular, D6', lambda: heisenberg(3, twod.D6(), T1, T2), None),
        ('2x2 two-orbital, D4', lambda: heisenberg(2, twod.D4(), orbital_mixing=orbital,
                                                   local_charge=(0, 1, 0, 1)), None),
    ]
    results = {'basis': True, 'action': True, 'momentum': True}
    for name, build, sztot in cases:
        model = build()
        print(f'── {name} ──')
        results['basis'] &= test_basis(model, sztot, name + ' basis')
        results['action'] &= test_action(model, sztot, name + ' action')
        results['momentum'] &= test_momentum(model, sztot, name + ' momentum basis')
        print()

    failed = [k for k, v in results.items() if not v]
    if failed:
        print('FAILED: ' + ', '.join(failed))
        sys.exit(1)
    print('The C++ state layer agrees with the Python one.')


if __name__ == '__main__':
    main()
