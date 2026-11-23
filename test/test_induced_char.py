'''
Test of the induced-character formula of doc/main.tex against the great
orthogonality theorem.

For a momentum *label* k (an integer tuple) with little group G_k, the irrep
``(k, α)`` of the lattice group ``G_latt = G_tr ⋊ G_pt`` is induced from
``χ_k ⊗ χ_α``; its character is

    χ_{k,α}(t, r) = Σ_{γ ∈ G_pt / G_k} χ_k(γ t) · χ̃_α(γ r γ^{-1}),

where χ̃_α is χ_α on G_k and zero outside, and γ runs over a transversal of
G_k in G_pt (``Model.little_group`` returns representatives of the right
cosets).  Only one momentum per star is induced, because k and r k give
equivalent representations.

The test verifies, for every lattice and point group in CASES,

    Σ_{g ∈ G_latt} χ_i(g) χ_j*(g) = |G_latt| δ_ij ,
    Σ_i d_i² = |G_latt| ,
    χ_i(e) = |G_pt / G_k| · d_α .
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import numpy as np

twod = importlib.import_module('geometry.2dsym')
from base import Model

A1 = np.array([1., 0.])
A2 = np.array([0., 1.])
T1 = np.array([1., 0.])
T2 = np.array([0.5, np.sqrt(3.) / 2.])


def build(N1, N2, point_group, a1=A1, a2=A2) -> Model:
    '''Empty lattice with the given point group.'''
    return Model(2, [(a1, N1), (a2, N2)], {('A', 0): np.array([0., 0.])},
                 point_group=point_group)


def star(model: Model, k: tuple[int, ...]) -> set[tuple[int, ...]]:
    '''Momenta equivalent to *k* under the point group.'''
    return {tuple((model.point_group.label_action(op, model.supercell_vectors).T
                   @ np.array(k)) % model.Nvec)
            for op in model.point_group}


def induced_character(model: Model, k: tuple[int, ...], alpha: str
                      ) -> np.ndarray:
    '''
    Induced character of the irrep ``(k, α)``, on all elements of G_latt.

    The elements are enumerated as ``(t, r)`` with ``t`` running over the
    translations and ``r`` over the point group; the character is evaluated at
    ``(t, r)`` with the formula quoted in the docstring of the module.
    '''
    lg, coset_reps = model.little_group(k)
    lg_ops = list(lg)
    ops = list(model.point_group)
    chi = np.zeros(len(model.tr_group) * len(ops), dtype=complex)
    for i, t in enumerate(model.tr_group):
        t_vec = model.tr_group.translation_vector(t)
        for j, r in enumerate(ops):
            value = 0j
            for gamma in coset_reps:
                # the rotated translation gamma t is again a translation
                rotated = model.tr_group.momentum_vector(
                    k, model.bz_cellvectors) @ (gamma @ t_vec)
                phase = np.exp(1j * rotated)
                conjugated = gamma @ r @ np.linalg.inv(gamma)
                if any(np.allclose(conjugated, h) for h in lg_ops):
                    value += phase * lg.character(alpha, conjugated)
            chi[i * len(ops) + j] = value
    return chi


def test_orthogonality(model: Model, label: str, tol: float = 1e-10,
                       verbose: bool = True) -> bool:
    '''
    Orthogonality, completeness and dimension of the induced characters.

    Args:
        model: the lattice.
        label: name used in the log.
        tol: absolute tolerance on the inner products.
        verbose: print the result.
    Returns:
        True if all checks passed.
    '''
    ops = list(model.point_group)
    order = len(model.tr_group) * len(ops)
    # one momentum per star
    reps = sorted({min(star(model, k)) for k in model.tr_group.all_momenta()})
    chars, dims = {}, {}
    for k in reps:
        lg, coset_reps = model.little_group(k)
        for alpha in lg.all_irrps():
            chars[(k, alpha)] = induced_character(model, k, alpha)
            dims[(k, alpha)] = len(coset_reps) * lg.irrep_dim(alpha)
    keys = list(chars)
    failures = []
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            inner = np.vdot(chars[ki], chars[kj]).real
            expected = order if i == j else 0.0
            if abs(inner - expected) > tol:
                failures.append(f'<{ki}|{kj}> = {inner:.6f} (expected {expected})')
    # dimensions: χ(e) = |G_pt / G_k| d_α, and Σ d² = |G_latt|
    for key in keys:
        identity_value = chars[key][0]
        if abs(identity_value - dims[key]) > tol:
            failures.append(f'χ_{key}(e) = {identity_value:.6f} but dim = {dims[key]}')
    if abs(sum(d ** 2 for d in dims.values()) - order) > tol:
        failures.append(f"Σ d² = {sum(d ** 2 for d in dims.values()):.6f} "
                        f"(expected |G_latt| = {order})")
    if verbose:
        print(f"  {'✓' if not failures else 'FAIL'}  {label}: {len(reps)} stars, "
              f"{len(keys)} induced irreps, |G_latt| = {order}, "
              f"Σ d² = {sum(d ** 2 for d in dims.values()):.0f}")
        for line in failures[:3]:
            print('       ' + line)
    return not failures


CASES = [
    ('2x2 square, D4', 2, 2, twod.D4(), A1, A2),
    ('2x2 square, C4', 2, 2, twod.C4(), A1, A2),
    ('2x1 rectangle, C2', 2, 1, twod.C2(), A1, A2),
    ('2x2 triangular, D3', 2, 2, twod.D3(), T1, T2),
    ('2x2 triangular, D6', 2, 2, twod.D6(), T1, T2),
    ('2x2 triangular, C6', 2, 2, twod.C6(), T1, T2),
    ('4x4 square, D4', 4, 4, twod.D4(), A1, A2),
]


def main():
    print('══════════════════════════════════════════════════════════')
    print('Induced-character orthogonality tests')
    print('══════════════════════════════════════════════════════════')
    print()
    ok = True
    for label, N1, N2, pg, a1, a2 in CASES:
        ok &= test_orthogonality(build(N1, N2, pg, a1, a2), label)
    print()
    if not ok:
        print('TEST FAILED.')
        sys.exit(1)
    print('All induced-character tests passed.')


if __name__ == '__main__':
    main()
