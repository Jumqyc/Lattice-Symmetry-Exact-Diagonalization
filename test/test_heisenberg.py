'''
The Heisenberg model on a 3x3 torus: every sector decomposition reproduces the
dense spectrum.

Three things are compared, each with fewer symmetries than the next:

* the dense Hamiltonian, assembled straight from the couplings, and diagonalised;
* the momentum sectors alone, without the point group;
* the full decomposition, with the point group and with the spin flip.

The first is the oracle and lives in ``reference.py``; it shares no code with the
library, so agreement is not a tautology.  The last two are built by the kernel.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib

import numpy as np

twod = importlib.import_module('geometry.2dsym')
try:
    import _edkern
except ImportError:                          # pragma: no cover
    _edkern = None
import kernel
import reference
from ed import Coupling, Model

X = np.array([[0., 1.], [1., 0.]], dtype=complex)
Y = np.array([[0., -1.], [1., 0.]], dtype=complex) * 1j
Z = np.array([[1., 0.], [0., -1.]], dtype=complex)


def build_heisenberg_model() -> Model:
    '''The 3x3 square-lattice Heisenberg model with D4 symmetry.'''
    model = Model(2, [(np.array([1., 0.]), 3), (np.array([0., 1.]), 3)],
                  {('A', 0): np.array([0., 0.])}, point_group=twod.D4())
    for i in range(3):
        for j in range(3):
            site = i * 3 + j
            right = i * 3 + (j + 1) % 3
            down = ((i + 1) % 3) * 3 + j
            model.add_coupling(Coupling(2, (site, right), (X, X), (Y, Y), (Z, Z)))
            model.add_coupling(Coupling(2, (site, down), (X, X), (Y, Y), (Z, Z)))
    return model


def momentum_spectrum(model: Model, sztot=None) -> np.ndarray:
    '''
    The union of the momentum sectors, without the point group.

    Args:
        model: the model.
        sztot: total charge, or None.
    Returns:
        The eigenvalues, ascending.
    '''
    basis = kernel.basis_of(model, sztot)
    hamiltonian = _edkern.assemble(basis, kernel.coupling_terms(model))
    translations = kernel.translations(model)
    values = []
    for k in model.tr_group.all_momenta():
        characters = [complex(model.tr_group.character(k, t)) for t in model.tr_group]
        columns = _edkern.momentum_basis(basis, translations, characters)
        if columns.columns() == 0:
            continue
        sector = _edkern.sector_hamiltonian(hamiltonian, columns)
        values.append(np.linalg.eigvalsh(kernel.to_scipy(sector).toarray()))
    return np.sort(np.concatenate(values))


def report(name: str, found: np.ndarray, want: np.ndarray, tol: float = 1e-10) -> bool:
    '''
    Compare two spectra and print the result.

    Args:
        name: label for the line.
        found: the spectrum under test.
        want: the reference spectrum.
        tol: tolerance on the largest difference.
    Returns:
        True if the spectra agree.
    '''
    if len(found) != len(want):
        print(f'  FAIL {name}: {len(found)} eigenvalues against {len(want)}')
        return False
    difference = float(np.max(np.abs(found - want)))
    print(f"  {'✓' if difference < tol else 'FAIL'}  {name}: {len(found)} eigenvalues, "
          f"max |ΔE| = {difference:.1e}")
    return difference < tol


def main() -> None:
    if _edkern is None:
        print('  skipped: the C++ kernel is not built (python cpp/build.py)')
        sys.exit(0)
    model = build_heisenberg_model()
    print(f'3x3 Heisenberg, D4: {model.size_of_unit_cell} sites, '
          f'|G_tr| = {len(model.tr_group)}, |G_pt| = {len(list(model.point_group))}')
    dense = reference.spectrum(reference.hamiltonian(model))
    print(f'  reference: {len(dense)} states, dense diagonalisation')

    ok = report('momentum sectors', momentum_spectrum(model), dense)
    ok &= report('full decomposition', reference.sector_spectrum(model.build_sectors()), dense)

    flipped = model.build_sectors(spin_flip=True)
    ok &= report('full decomposition + spin flip', reference.sector_spectrum(flipped), dense)
    dimensions = sum(sector.dim for sector in flipped.values())
    ok &= dimensions == len(dense)
    print(f"  {'✓' if dimensions == len(dense) else 'FAIL'}  dimensions add up: "
          f"{dimensions}/{len(dense)}")

    print('Test passed.' if ok else 'TEST FAILED.')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
