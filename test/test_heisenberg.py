'''
Test _build_H on a 3×3 Heisenberg model against direct diagonalization.

Implements in order: Step 1 (momentum), Step 3 (spin-flip), Step 2 (little group).
After each step we compare the union of all sector eigenvalues with the
eigenvalues from a direct (full-Hilbert-space) sparse diagonalization.
'''

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Optional
import importlib

twod = importlib.import_module('geometry.2dsym')
from ed import Coupling, Model

# ══════════════════════════════════════════════════════════════════════════════
# Pauli matrices
# ══════════════════════════════════════════════════════════════════════════════
X = np.array([[0., 1.], [1., 0.]], dtype=complex)
Y = np.array([[0., -1j], [1j, 0.]], dtype=complex)
Z = np.array([[1., 0.], [0., -1.]], dtype=complex)

# ══════════════════════════════════════════════════════════════════════════════
# Build the 3×3 square-lattice Heisenberg model
# ══════════════════════════════════════════════════════════════════════════════

def build_heisenberg_model() -> Model:
    '''3×3 square lattice with PBC.  C₂ point group (π rotation).'''
    m = Model(
        phys_dim=2,
        supercell_info=[(np.array([1., 0.]), 3),
                        (np.array([0., 1.]), 3)],
        cell_info={('A', 0): np.array([0., 0.])},
        point_group=twod.C2(),
    )
    # Heisenberg bonds: horizontal and vertical neighbours with PBC wrap
    for i in range(3):
        for j in range(3):
            site = i * 3 + j
            right = i * 3 + (j + 1) % 3
            down  = ((i + 1) % 3) * 3 + j
            if site < right:
                m.add_coupling(Coupling(2, (site, right), (X, X), (Y, Y), (Z, Z)))
            if site < down:
                m.add_coupling(Coupling(2, (site, down),  (X, X), (Y, Y), (Z, Z)))
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Direct diagonalization (baseline)
# ══════════════════════════════════════════════════════════════════════════════

def direct_diag(model: Model, Sztot: Optional[int] = None) -> np.ndarray:
    '''Build the symmetrized H and return eigenvalues.'''
    H_sym = model._build_H_sym(Sztot).toarray()
    return np.linalg.eigvalsh(H_sym)


# ══════════════════════════════════════════════════════════════════════════════
# Test helper
# ══════════════════════════════════════════════════════════════════════════════

def compare_eigenvalues(name: str, sectors: dict, direct_ev: np.ndarray,
                        tol: float = 1e-8):
    '''Collect eigenvalues from all sectors, sort, compare with direct.'''
    all_ev = np.sort(np.concatenate(
        [np.linalg.eigvalsh(H.toarray()) for H in sectors.values()]))
    direct_sorted = np.sort(direct_ev)
    if len(all_ev) != len(direct_sorted):
        print(f'  FAIL {name}: dim mismatch {len(all_ev)} vs {len(direct_sorted)}')
        return False
    diff = np.max(np.abs(all_ev - direct_sorted))
    if diff > tol:
        print(f'  FAIL {name}: max |ΔE| = {diff:.2e}')
        return False
    print(f'  ✓  {name}: {len(all_ev)} eigenvalues match (max |ΔE| = {diff:.2e})')
    return True


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('Building 3×3 Heisenberg model (C₂ point group) …')
    model = build_heisenberg_model()
    print(f'  sites={model.size_of_unit_cell}, |G_tr|={len(model.tr_group)}, '
          f'|G_pt|={len(list(model.point_group))}')
    print()

    # Use full Hilbert space — spin-flip is a symmetry here
    print('Direct diagonalization (full space, 512 states) …')
    direct_ev = direct_diag(model, None)
    print(f'  {len(direct_ev)} eigenvalues computed')
    print()

    # ── Momentum-sector Hamiltonian (Step 1) ───────────────────────────
    print('Momentum projection (Step 1) —──')
    all_sectors = model._build_H(None)
    ok = compare_eigenvalues('momentum', all_sectors, direct_ev)
    print()

    if ok:
        print('Test passed.')
    else:
        print('TEST FAILED.')
        sys.exit(1)


if __name__ == '__main__':
    main()
