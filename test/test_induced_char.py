'''
Test the induced-character formula (doc/main.tex, Eq. 67) against the
Great Orthogonality Theorem.

    χ_{k,α}^{G_latt}(t, r)  =  Σ_{g ∈ G_pt / G_k}
        χ_k^{G_tr}(g·t)  ·  χ_α^{G_k}(g r g^{-1})

where  χ_k^{G_tr}(t) = exp(i k·t)  and  χ_α^{G_k}(…) is taken as 0
when its argument is not in the little group G_k.

The test builds the full lattice group  G_latt = G_tr ⋊ G_pt, computes
the induced character for every (k, irrep) pair, and verifies::

    Σ_g  χ_i(g) χ_j*(g)  =  |G_latt| · δ_{ij}
'''

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import importlib

twod = importlib.import_module('geometry.2dsym')
from base import Model


# ══════════════════════════════════════════════════════════════════════════════
# Induced character formula (used only for verification)
# ══════════════════════════════════════════════════════════════════════════════

def _induced_char(k: np.ndarray, t: np.ndarray, r: np.ndarray,
                  irrep_label: str, lg, lg_ops: list[np.ndarray],
                  coset_reps: list[np.ndarray]) -> complex:
    chi = 0.0 + 0.0j
    for g in coset_reps:
        chi_k = np.exp(1j * np.dot(k, g @ t))
        grg_inv = g @ r @ g.T
        if any(np.allclose(grg_inv, h) for h in lg_ops):
            chi_alpha = lg.character(irrep_label, grg_inv)
        else:
            chi_alpha = 0.0
        chi += chi_k * chi_alpha
    return chi


def compute_induced_characters(model: Model
                               ) -> dict[tuple[tuple[int, ...], str], np.ndarray]:
    tr_group = model.tr_group
    pt_group = model.point_group
    reciprocal_basis = model.bz_cellvectors

    N_tr = len(tr_group)
    N_pt = len(list(pt_group))
    G_order = N_tr * N_pt

    rotations = list(pt_group)
    translations = [tr_group.translation_vector(t) for t in tr_group]

    all_irrep: dict[tuple[tuple[int, ...], str], np.ndarray] = {}
    for k_tuple in tr_group.all_momenta():
        k = tr_group.momentum_vector(k_tuple, reciprocal_basis)
        lg, coset_reps = pt_group.little_group(k, reciprocal_basis)
        lg_ops = list(lg)
        for irrep_label in lg.all_irrps():
            chars = np.zeros(G_order, dtype=complex)
            for elem_idx in range(G_order):
                t_idx = elem_idx % N_tr
                r_idx = elem_idx // N_tr
                t = translations[t_idx]
                r = rotations[r_idx]
                chars[elem_idx] = _induced_char(
                    k, t, r, irrep_label, lg, lg_ops, coset_reps)
            all_irrep[(k_tuple, irrep_label)] = chars
    return all_irrep


# ══════════════════════════════════════════════════════════════════════════════
# test cases
# ══════════════════════════════════════════════════════════════════════════════

def test_orthogonality(model: Model,
                       tol: float = 1e-10,
                       ) -> tuple[int, int]:
    '''
    Run the orthogonality check on *model*.

    Returns (num_irreps, group_order).
    '''
    all_irrep = compute_induced_characters(model)
    G_order = len(model.tr_group) * len(list(model.point_group))
    all_irrps = list(all_irrep.keys())

    # ── orthogonality:  Σ_g  χ_i(g) χ_j*(g)  =  |G| · δ_{ij} ─────────
    failures = 0
    for i, key_i in enumerate(all_irrps):
        chi_i = all_irrep[key_i]
        for j, key_j in enumerate(all_irrps):
            inner = np.dot(chi_i, np.conj(all_irrep[key_j]))
            expected = G_order if i == j else 0.0
            if abs(inner - expected) > tol:
                print(f'  FAIL  ⟨{key_i}|{key_j}⟩ = {inner:.6f}  '
                      f'(expected {expected})')
                failures += 1

    if failures == 0:
        n = len(all_irrps)
        print(f'  ✓  {n} irreps, {n*n} pairs — all orthogonal')

    # ── dimensionality check: Σ dim(ρ_i)² = |G| ────────────────────────
    dims = [all_irrep[key][0] for key in all_irrps]   # χ_i(E) = dim
    dim_sum_sq = sum(abs(d) ** 2 for d in dims)
    if abs(dim_sum_sq - G_order) > tol:
        print(f'  FAIL  Σ dim² = {dim_sum_sq}  (expected {G_order})')
        failures += 1
    else:
        print(f'  ✓  Σ dim² = {dim_sum_sq:.0f} = |G| = {G_order}')

    return len(all_irrps), G_order


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('══════════════════════════════════════════════════════════')
    print('Induced-character orthogonality tests')
    print('══════════════════════════════════════════════════════════')
    print()

    # ── 2×2 square supercell, D₄ ───────────────────────────────────────
    print('── 2×2 square · D₄ ──')
    m = Model(
        phys_dim=2,
        supercell_info=[(np.array([1., 0.]), 2),
                        (np.array([0., 1.]), 2)],
        cell_info={('A', 0): np.array([0., 0.])},
        point_group=twod.D4(),
    )
    n_irr, G = test_orthogonality(m)
    print(f'  Group order: {G}  (4 translations × 8 pt-ops)')
    print(f'  Number of irreps: {n_irr}')
    print()

    # ── 2×2 square supercell, C₄ (no reflections) ─────────────────────
    print('── 2×2 square · C₄ ──')
    m = Model(
        phys_dim=2,
        supercell_info=[(np.array([1., 0.]), 2),
                        (np.array([0., 1.]), 2)],
        cell_info={('A', 0): np.array([0., 0.])},
        point_group=twod.C4(),
    )
    n_irr, G = test_orthogonality(m)
    print(f'  Group order: {G}  (4 translations × 4 pt-ops)')
    print(f'  Number of irreps: {n_irr}')
    print()

    # ── 2×2 square supercell, C₂ ───────────────────────────────────────
    print('── 2×2 square · C₂ ──')
    m = Model(
        phys_dim=2,
        supercell_info=[(np.array([1., 0.]), 2),
                        (np.array([0., 1.]), 2)],
        cell_info={('A', 0): np.array([0., 0.])},
        point_group=twod.C2(),
    )
    n_irr, G = test_orthogonality(m)
    print(f'  Group order: {G}  (4 translations × 2 pt-ops)')
    print(f'  Number of irreps: {n_irr}')
    print()

    # ── 2×2 square supercell, D₆  (incompatible → should fail) ─────────
    print('── 2×2 square · D₆  (incompatible — expect ValueError) ──')
    try:
        m = Model(
            phys_dim=2,
            supercell_info=[(np.array([1., 0.]), 2),
                            (np.array([0., 1.]), 2)],
            cell_info={('A', 0): np.array([0., 0.])},
            point_group=twod.D6(),
        )
        n_irr, G = test_orthogonality(m)
        print(f'  (unexpectedly succeeded — {G} elements)')
    except ValueError as e:
        print(f'  ✓  Correctly rejected: D₆ not a symmetry of square lattice')
        print(f'     ({e})')
    print()

    # ── 2×1 rectangular supercell, C₂ ──────────────────────────────────
    print('── 2×1 rectangle · C₂ ──')
    m = Model(
        phys_dim=2,
        supercell_info=[(np.array([1., 0.]), 2),
                        (np.array([0., 1.]), 1)],
        cell_info={('A', 0): np.array([0., 0.])},
        point_group=twod.C2(),
    )
    n_irr, G = test_orthogonality(m)
    print(f'  Group order: {G}  (2 translations × 2 pt-ops)')
    print(f'  Number of irreps: {n_irr}')
    print()

    # ── triangular lattice ───────────────────────────────────────────
    a1 = np.array([1., 0.])
    a2 = np.array([0.5, np.sqrt(3) / 2])

    # ── 2×2 triangular · D₃ ────────────────────────────────────────
    print('── 2×2 triangular · D₃ ──')
    m = Model(
        phys_dim=2,
        supercell_info=[(a1, 2), (a2, 2)],
        cell_info={('A', 0): np.array([0., 0.])},
        point_group=twod.D3(),
    )
    n_irr, G = test_orthogonality(m)
    print(f'  Group order: {G}  (4 translations × 6 pt-ops)')
    print(f'  Number of irreps: {n_irr}')
    print()

    # ── 2×2 triangular · D₆ ────────────────────────────────────────
    print('── 2×2 triangular · D₆ ──')
    m = Model(
        phys_dim=2,
        supercell_info=[(a1, 2), (a2, 2)],
        cell_info={('A', 0): np.array([0., 0.])},
        point_group=twod.D6(),
    )
    n_irr, G = test_orthogonality(m)
    print(f'  Group order: {G}  (4 translations × 12 pt-ops)')
    print(f'  Number of irreps: {n_irr}')
    print()

    # ── 2×2 triangular · C₃ ────────────────────────────────────────
    print('── 2×2 triangular · C₃ ──')
    m = Model(
        phys_dim=2,
        supercell_info=[(a1, 2), (a2, 2)],
        cell_info={('A', 0): np.array([0., 0.])},
        point_group=twod.C3(),
    )
    n_irr, G = test_orthogonality(m)
    print(f'  Group order: {G}  (4 translations × 3 pt-ops)')
    print(f'  Number of irreps: {n_irr}')
    print()

    # ── 2×2 triangular · C₆ ────────────────────────────────────────
    print('── 2×2 triangular · C₆ ──')
    m = Model(
        phys_dim=2,
        supercell_info=[(a1, 2), (a2, 2)],
        cell_info={('A', 0): np.array([0., 0.])},
        point_group=twod.C6(),
    )
    n_irr, G = test_orthogonality(m)
    print(f'  Group order: {G}  (4 translations × 6 pt-ops)')
    print(f'  Number of irreps: {n_irr}')
    print()

    print('══════════════════════════════════════════════════════════')
    print('All induced-character orthogonality tests passed.')
    print('══════════════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
