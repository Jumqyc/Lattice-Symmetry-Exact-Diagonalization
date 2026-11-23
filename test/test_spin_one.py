'''
Spin one: the AKLT ring, where the ground state is known in closed form.

The chain is the first model here whose local dimension is not two, and its ring
is the first whose point group is not a piece of D2 acting faithfully -- the ring
reflects in one mirror only.  What is checked is what the VBS state guarantees:
every bond projector annihilates it, so the ground energy is zero; the ring has no
edges, so the ground state is unique; and it is a singlet, so <S^2> is zero and the
state sits in the totally symmetric sector at zero momentum.
'''

from __future__ import annotations

import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reference  # noqa: E402
from ed import Coupling, Model  # noqa: E402
from representation import ExplicitPointGroup, TrivialGroup  # noqa: E402

twod = importlib.import_module('geometry.2dsym')

SPIN = 1.0
A1, A2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])


def spin_one():
    '''S^x, S^y, S^z for spin one, in the m = -1, 0, 1 ordering.'''
    m = np.arange(3) - 1.0
    rising, lowering = np.zeros((3, 3)), np.zeros((3, 3))
    for i, value in enumerate(m[:-1]):
        rising[i + 1, i] = np.sqrt(SPIN * (SPIN + 1) - value * (value + 1))
    for i, value in enumerate(m[1:], start=1):
        lowering[i - 1, i] = np.sqrt(SPIN * (SPIN + 1) - value * (value - 1))
    return (rising + lowering) / 2, (rising - lowering) / 2j, np.diag(m)


def ring(length: int, periodic: bool) -> Model:
    '''
    The AKLT chain, H = sum_bonds P_2 with P_2 the projector on total spin two.

    Args:
        length: number of spin-one sites.
        periodic: whether the last bond closes the ring.
    Returns:
        The model.  The ring is given one cell per site and the point group of the
        chain -- the identity and the reflection along it -- because a point group
        whose action on the sites has a kernel is not supported; the open chain
        keeps every site in one cell and takes the trivial group.
    '''
    operators = spin_one()
    if periodic:
        cells = [(A1, length), (A2, 1)]
        orbitals = {('A', 0): np.array([0.0, 0.0])}
        point_group = ExplicitPointGroup([np.eye(2), np.diag([-1.0, 1.0])])
    else:
        cells = [(A1 * length, 1), (A2, 1)]
        orbitals = {('A', i): np.array([float(i), 0.0]) for i in range(length)}
        point_group = TrivialGroup()
    model = Model(3, cells, orbitals, point_group=point_group)
    # P_2 = 1/3 + (1/2) S.S + (1/6) (S.S)^2, and (S.S)^2 = sum_{a,b} (S_a S_b)(x)(S_a S_b)
    products = [(np.eye(3) / 3.0, np.eye(3))]
    products += [(0.5 * a, a) for a in operators]
    products += [((a @ b) / 6.0, a @ b) for a in operators for b in operators]
    bonds = [(i, i + 1) for i in range(length - 1)]
    if periodic:
        bonds.append((length - 1, 0))
    for i, j in bonds:
        model.add_coupling(Coupling(3, (i, j), *products))
    return model


def spin_squared(length: int) -> np.ndarray:
    '''
    The dense S^2 = (sum_i S_i)^2 of a chain of *length* spin-one sites.

    Args:
        length: number of sites.
    Returns:
        A dense matrix on the whole space, in the basis order of the package.
    '''
    operators = spin_one()
    model = Model(3, [(A1 * length, 1), (A2, 1)],
                  {('A', i): np.array([float(i), 0.0]) for i in range(length)},
                  point_group=TrivialGroup())
    for i in range(length):
        model.add_coupling(Coupling(3, (i,), *[(a @ a,) for a in operators]))
    for i in range(length):
        for j in range(i + 1, length):
            model.add_coupling(Coupling(3, (i, j), *[(2 * a, a) for a in operators]))
    return reference.hamiltonian(model, None)


def test_ring(length: int = 6) -> bool:
    '''
    The bulk ground state of the ring: unique, at zero energy, and a singlet.

    Args:
        length: number of sites.
    Returns:
        Whether every check passed.
    '''
    model = ring(length, periodic=True)
    sectors = model.build_sectors()
    total = 3 ** length
    sizes = {key: sector.hamiltonian.shape[0] for key, sector in sectors.items()}
    complete = sum(sizes.values()) == total

    squared = spin_squared(length)
    grounds = []
    for key, sector in sorted(sectors.items(), key=str):
        values, columns = np.linalg.eigh(sector.hamiltonian.toarray())
        if values[0] > 1e-9:
            continue
        state = np.asarray(sector.basis @ columns[:, 0]).ravel()
        grounds.append((key, float(values[0]),
                        float(np.vdot(state, squared @ state).real / np.vdot(state, state).real)))
    unique = len(grounds) == 1
    key, energy, spin = grounds[0] if grounds else ((None, None, None), 0.0, 0.0)[1:]
    symmetric = str(key[0]) == '(0, 0)' and str(key[1]).startswith('A1')
    ok = complete and unique and abs(energy) < 1e-9 and abs(spin) < 1e-9 and symmetric

    exact = reference.spectrum(reference.hamiltonian(model, None))
    gap = float(np.sort(exact)[len(grounds)] - np.sort(exact)[0]) if grounds else 0.0
    print(f"  {'✓' if ok else 'FAIL'}  AKLT ring L={length}: {len(sectors)} sectors "
          f"summing to {sum(sizes.values())} of {total}, ground state "
          f"{'unique' if unique else len(grounds)} in {key} with E {energy:.1e} and "
          f"<S^2> {spin:.1e}, bulk gap {gap:.4f}")
    return ok


def test_open_chain(length: int = 4) -> bool:
    '''
    The open chain: zero energy, and the edge spins give four ground states.

    Args:
        length: number of sites.
    Returns:
        Whether every check passed.
    '''
    model = ring(length, periodic=False)
    values = np.sort(reference.spectrum(reference.hamiltonian(model, None)))
    degenerate = int(np.sum(np.abs(values[:4] - values[0]) < 1e-9))
    ok = abs(values[0]) < 1e-9 and degenerate == 4
    print(f"  {'✓' if ok else 'FAIL'}  AKLT open chain L={length}: lowest {values[0]:.1e}, "
          f"ground degeneracy {degenerate} (two free spin halves), next level "
          f"{values[4]:.4f}")
    return ok


def main() -> int:
    '''Run the checks and report.'''
    results = [test_ring(6), test_open_chain(4)]
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
