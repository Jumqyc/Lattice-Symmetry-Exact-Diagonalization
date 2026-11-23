'''
Tests for the point-group part of the symmetry machinery.

Three things are checked:

1. **Little group.**  For every momentum label the star is computed from the
   action of the point group on the labels, and the orbit–stabiliser theorem
   ``|Star(k)| * |G_k| = |G_pt|`` must hold.  This is what fails when the
   little-group criterion is applied to the geometric momentum instead of to
   the integer label.

2. **Character tables.**  The generic character table of an explicitly given
   group (the class used for little groups, which are generic subgroups) must
   agree with the hand-written tables of the named groups, and both must obey
   the orthogonality relations.

3. **Sectors.**  The sector decomposition is compared with direct
   diagonalisation, the dimensions are recomputed from characters, and every
   sector vector is tested against the full-space symmetry operators:
   momentum ``χ_k(t)``, point-group irrep ``Π_α``, spin-flip parity ``s``.

Everything is checked for the Heisenberg model, which commutes with all three
operations.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
from itertools import product

import numpy as np

twod = importlib.import_module('geometry.2dsym')
from ed import Coupling, Model
from lattice_action import label_action
from representation import ExplicitPointGroup, TrivialGroup

X = np.array([[0., 1.], [1., 0.]], dtype=complex)
Y = np.array([[0., -1j], [1j, 0.]], dtype=complex)
Z = np.array([[1., 0.], [0., -1.]], dtype=complex)

A1 = np.array([1., 0.])
A2 = np.array([0., 1.])
T1 = np.array([1., 0.])
T2 = np.array([0.5, np.sqrt(3.) / 2.])


# ══════════════════════════════════════════════════════════════════════════════
# models
# ══════════════════════════════════════════════════════════════════════════════

def heisenberg(N1, N2, point_group, a1=A1, a2=A2) -> Model:
    '''Nearest-neighbour Heisenberg model on an N1 x N2 torus.'''
    m = Model(2, [(a1, N1), (a2, N2)], {('A', 0): np.array([0., 0.])},
              point_group=point_group)
    for i in range(N1):
        for j in range(N2):
            site = i * N2 + j
            right = i * N2 + (j + 1) % N2
            down = ((i + 1) % N1) * N2 + j
            if site < right:
                m.add_coupling(Coupling(2, (site, right), (X, X), (Y, Y), (Z, Z)))
            if site < down:
                m.add_coupling(Coupling(2, (site, down), (X, X), (Y, Y), (Z, Z)))
    return m


CASES = [
    ('3x3 square, D4', 3, 3, twod.D4(), A1, A2),
    ('3x3 square, C4', 3, 3, twod.C4(), A1, A2),
    ('3x3 triangular, D6', 3, 3, twod.D6(), T1, T2),
    ('3x3 triangular, D3', 3, 3, twod.D3(), T1, T2),
    ('2x4 rectangle, C2', 2, 4, twod.C2(), A1, A2),
]


# ══════════════════════════════════════════════════════════════════════════════
# full-space symmetry operators
# ══════════════════════════════════════════════════════════════════════════════

def state_map_point_group(model: Model, op: np.ndarray):
    '''Action of a point-group operation on a computational basis state.'''
    perm = model._op_perm(op)
    return lambda state: tuple(perm(np.array(state)))


def state_map_translation(model: Model, t):
    '''Action of a translation on a computational basis state.'''
    perm = model._permutation(t, 0)
    return lambda state: tuple(perm(np.array(state)))


def state_map_flip(model: Model):
    '''Action of the spin flip on a computational basis state.'''
    d = model.physical_dim
    return lambda state: tuple(d - 1 - s for s in state)


def apply_matrix(model: Model, state_map, states, s2i, v: np.ndarray) -> np.ndarray:
    '''
    Apply a permutation operator to a dense vector.

    ``(U v)[state_map(s)] = v[s]``, which is how a site permutation acts on the
    computational basis.
    '''
    out = np.zeros_like(v)
    for i, state in enumerate(states):
        out[s2i[state_map(state)]] += v[i]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1. little groups
# ══════════════════════════════════════════════════════════════════════════════

def test_little_groups(verbose: bool = True) -> bool:
    '''Orbit-stabiliser theorem for every momentum of every test case.'''
    ok = True
    for label, N1, N2, pg, a1, a2 in CASES:
        m = heisenberg(N1, N2, pg, a1, a2)
        ops = list(pg)
        bad = []
        for k in m.tr_group.all_momenta():
            star = {tuple((label_action(op, m.supercell_vectors).T @ np.array(k))
                          % m.Nvec) for op in ops}
            lg, reps = m.little_group(k)
            if len(star) * len(list(lg)) != len(ops):
                bad.append((k, len(star), len(list(lg))))
            if len(reps) * len(list(lg)) != len(ops):
                bad.append((k, 'cosets', len(reps), len(list(lg))))
        ok &= not bad
        if verbose:
            sizes = sorted({len(list(m.little_group(k)[0]))
                            for k in m.tr_group.all_momenta()})
            print(f"  {'✓' if not bad else 'FAIL'}  {label}: little-group sizes "
                  f"{sizes} obey |Star|·|G_k| = {len(ops)}"
                  + ('' if not bad else f'  first failure {bad[0]}'))
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 2. character tables
# ══════════════════════════════════════════════════════════════════════════════

def test_character_tables(tol: float = 1e-10, verbose: bool = True) -> bool:
    '''Generic tables agree with the hand-written ones and are orthogonal.'''
    ok = True
    for pg in [twod.C2(), twod.C3(), twod.C4(), twod.C6(),
               twod.D3(), twod.D4(), twod.D6()]:
        ops = list(pg)
        generic = ExplicitPointGroup(ops)
        named = np.array([[pg.character(a, op) for op in ops] for a in pg.all_irrps()])
        auto = np.array([[generic.character(a, op) for op in ops]
                         for a in generic.all_irrps()])
        # same set of rows, possibly under another name
        same = (len(named) == len(auto)
                and all(any(np.allclose(row, other) for other in auto) for row in named))
        gram = auto @ auto.conj().T
        ortho = np.allclose(gram, len(ops) * np.eye(len(auto)), atol=tol)
        dims = sum(generic.irrep_dim(a) ** 2 for a in generic.all_irrps())
        sumsq = dims == len(ops)
        ok &= same and ortho and sumsq
        if verbose:
            print(f"  {'✓' if same and ortho and sumsq else 'FAIL'}  "
                  f"{type(pg).__name__}: generic table = named table {same}, "
                  f"orthogonal {ortho}, Σd² = {dims} = |G| {sumsq}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 3. sectors
# ══════════════════════════════════════════════════════════════════════════════

def test_sector_spectra(verbose: bool = True, tol: float = 1e-9) -> bool:
    '''Union of the sector spectra is the full spectrum, and Σ dim = dim H.'''
    ok = True
    for label, N1, N2, pg, a1, a2 in CASES:
        m = heisenberg(N1, N2, pg, a1, a2)
        direct = np.sort(np.linalg.eigvalsh(m._build_H_sym().toarray()))
        n_states = len(direct)
        for spin_flip in (False, True):
            sectors = m.build_sectors(spin_flip=spin_flip)
            ev = np.sort(np.concatenate(
                [np.linalg.eigvalsh(s.hamiltonian.toarray()) for s in sectors.values()]))
            dim = sum(s.dim for s in sectors.values())
            good = (len(ev) == n_states and dim == n_states
                    and np.max(np.abs(ev - direct)) < tol)
            ok &= good
            if verbose:
                print(f"  {'✓' if good else 'FAIL'}  {label}"
                      f"{' + F' if spin_flip else ''}: {len(sectors)} sectors, "
                      f"Σ dim = {dim}/{n_states}, "
                      f"max |ΔE| = {np.max(np.abs(ev - direct)):.1e}")
    return ok


def test_sector_labels(tol: float = 1e-8, verbose: bool = True) -> bool:
    '''
    Every sector vector carries the labels that the sector claims.

    For each column v of a sector basis: ``<v|U(t)|v> = χ_k(t)`` for every
    translation, ``<v|U(r)|v> = χ_α(r)`` for every r in the little group (only
    for one-dimensional α), ``Π_α v = v`` and ``Π_β v = 0`` for β ≠ α, and
    ``<v|F|v> = s`` for the spin-flip parity.
    '''
    ok = True
    for label, N1, N2, pg, a1, a2 in CASES:
        m = heisenberg(N1, N2, pg, a1, a2)
        states, s2i = m._hilbert_space()
        sectors = m.build_sectors(spin_flip=True)
        worst = 0.0
        for key, sector in sectors.items():
            k, alpha, s = key
            lg, _ = m.little_group(k)
            ops = list(lg)
            for column in range(sector.dim):
                v = np.asarray(sector.basis[:, column].todense()).ravel()
                # momentum
                for t in m.tr_group:
                    overlap = np.vdot(v, apply_matrix(
                        m, state_map_translation(m, t), states, s2i, v))
                    worst = max(worst, abs(overlap - m.tr_group.character(k, t)))
                # point group: one-dimensional irreps are diagonal on the sector
                if alpha is not None:
                    for op in ops:
                        if lg.irrep_dim(alpha) != 1:
                            continue
                        overlap = np.vdot(v, apply_matrix(
                            m, state_map_point_group(m, op), states, s2i, v))
                        worst = max(worst, abs(overlap - lg.character(alpha, op)))
                    if lg.irrep_dim(alpha) > 1:
                        for beta in lg.all_irrps():
                            u = np.zeros_like(v)
                            for op in ops:
                                u += np.conj(lg.character(beta, op)) * apply_matrix(
                                    m, state_map_point_group(m, op), states, s2i, v)
                            u *= lg.irrep_dim(beta) / len(ops)
                            worst = max(worst, np.linalg.norm(
                                u - (v if beta == alpha else 0.0)))
                # spin flip
                overlap = np.vdot(v, apply_matrix(
                    m, state_map_flip(m), states, s2i, v))
                worst = max(worst, abs(overlap - s))
        good = worst < tol
        ok &= good
        if verbose:
            print(f"  {'✓' if good else 'FAIL'}  {label}: sector labels hold, "
                  f"max deviation = {worst:.1e}")
    return ok


def test_dimensions_from_characters(tol: float = 1e-8, verbose: bool = True) -> bool:
    '''
    Sector dimensions recomputed from characters.

    ``dim H_{k,α} = (d_α/|G_k|) Σ_r χ_α*(r) Tr_{H_k}(U_r)``, where the trace is
    taken over the momentum sector only; this is independent of how the
    isotypic basis is built and catches a wrong little group or a dropped
    Bloch phase.
    '''
    ok = True
    for label, N1, N2, pg, a1, a2 in CASES:
        m = heisenberg(N1, N2, pg, a1, a2)
        states, s2i = m._hilbert_space()
        sectors = m.build_sectors()
        for k in m.tr_group.all_momenta():
            lg, _ = m.little_group(k)
            ops = list(lg)
            reps, B, state2col = m._momentum_basis(k, states, s2i)
            traces = {}
            for op in ops:
                act, phase = m._sector_action(
                    state_map_point_group(m, op), states, s2i, state2col, B)
                traces[tuple(np.round(op.ravel(), 8))] = float(np.real(np.sum(
                    phase * (act == np.arange(len(act))))))
            for alpha in (['A'] if len(ops) == 1 else lg.all_irrps()):
                d = 1 if len(ops) == 1 else lg.irrep_dim(alpha)
                expected = 0.0
                for op in ops:
                    chi = 1.0 if len(ops) == 1 else lg.character(alpha, op)
                    expected += np.conj(chi) * traces[tuple(np.round(op.ravel(), 8))]
                expected = float(np.real(expected) * d / len(ops))
                found = sectors[(k, alpha, None)].dim if (k, alpha, None) in sectors else 0
                if abs(expected - found) > tol:
                    ok = False
                    if verbose:
                        print(f"  FAIL  {label}: dim (k={k}, α={alpha}) "
                              f"= {found} but {expected:.6f} from characters")
        if verbose and ok:
            print(f"  ✓  {label}: sector dimensions match the character formula")
    return ok


def test_rejects_bad_groups(verbose: bool = True) -> bool:
    '''Operations that are not symmetries of the torus must be refused.'''
    ok = True
    checks = [
        ('C4 on a 4x2 torus', lambda: Model(2, [(A1, 4), (A2, 2)],
                                            {('A', 0): np.array([0., 0.])},
                                            point_group=twod.C4())),
        ('D6 on a square lattice', lambda: Model(2, [(A1, 3), (A2, 3)],
                                                 {('A', 0): np.array([0., 0.])},
                                                 point_group=twod.D6())),
        ('C3 minus C3^2', lambda: Model(2, [(A1, 3), (A2, 3)],
                                        {('A', 0): np.array([0., 0.])},
                                        point_group=ExplicitPointGroup(
                                            [np.eye(2), twod.rotation(2 * np.pi / 3)]))),
    ]
    for label, build in checks:
        try:
            build()
        except ValueError as error:
            if verbose:
                print(f"  ✓  {label}: refused ({str(error)[:60]}…)")
        else:
            ok = False
            if verbose:
                print(f"  FAIL  {label}: accepted, but it is not a symmetry")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    results = {}
    for name, fn in [('little groups', test_little_groups),
                     ('character tables', test_character_tables),
                     ('bad groups refused', test_rejects_bad_groups),
                     ('sector spectra', test_sector_spectra),
                     ('sector dimensions', test_dimensions_from_characters),
                     ('sector labels', test_sector_labels)]:
        print(f'── {name} ──')
        results[name] = fn()
        print()
    failed = [k for k, v in results.items() if not v]
    if failed:
        print('FAILED: ' + ', '.join(failed))
        sys.exit(1)
    print('All point-group tests passed.')


if __name__ == '__main__':
    main()
