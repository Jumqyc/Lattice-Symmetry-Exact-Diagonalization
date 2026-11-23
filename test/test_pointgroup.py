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
try:
    import _edkern
except ImportError:                          # pragma: no cover
    _edkern = None
import kernel
import reference
from ed import Coupling, Model
from hilbert_action import circle_doublet
from lattice_action import AffineAction, label_action
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
    # a1 - a2 is a nearest-neighbour direction on the triangular lattice, and the
    # 60-degree rotation maps an a1 bond onto it
    directions = [(0, 1), (1, 0)]
    if np.allclose(np.linalg.norm(np.asarray(a1) - np.asarray(a2)), np.linalg.norm(a1)):
        directions.append((1, -1))
    for i in range(N1):
        for j in range(N2):
            site = i * N2 + j
            # the wrap bond is the instance j = N2 - 1 (i = N1 - 1)
            for di, dj in directions:
                partner = ((i + di) % N1) * N2 + (j + dj) % N2
                m.add_coupling(Coupling(2, (site, partner), (X, X), (Y, Y), (Z, Z)))
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

def action_point_group(model: Model, op: np.ndarray):
    '''Action of a point-group operation on the computational basis.'''
    return model._action_for(op)


def action_translation(model: Model, t):
    '''Action of a translation on the computational basis.'''
    perm = model._permutation(t, 0)
    return lambda state: tuple(perm(np.array(state)))


def action_flip(model: Model):
    '''Action of the spin flip on a computational basis state.'''
    d = model.physical_dim
    return lambda state: tuple(d - 1 - s for s in state)


def apply_operator(model: Model, state_map, states, s2i, v: np.ndarray) -> np.ndarray:
    '''
    Apply a symmetry operator to a dense vector.

    ``state_map`` is either a function ``state -> state`` or an object with an
    ``on_state`` method returning ``(state, phase)``.
    '''
    out = np.zeros_like(v)
    for i, state in enumerate(states):
        if hasattr(state_map, 'on_state'):
            image, phase = state_map.on_state(state)
            out[s2i[image]] += phase * v[i]
        else:
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
        direct = np.sort(reference.spectrum(reference.hamiltonian(m)))
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
        states = reference.basis_states(m)
        s2i = {state: row for row, state in enumerate(states)}
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
                    overlap = np.vdot(v, apply_operator(
                        m, action_translation(m, t), states, s2i, v))
                    worst = max(worst, abs(overlap - m.tr_group.character(k, t)))
                # point group: one-dimensional irreps are diagonal on the sector
                if alpha is not None:
                    for op in ops:
                        if lg.irrep_dim(alpha) != 1:
                            continue
                        overlap = np.vdot(v, apply_operator(
                            m, action_point_group(m, op), states, s2i, v))
                        worst = max(worst, abs(overlap - lg.character(alpha, op)))
                    if lg.irrep_dim(alpha) > 1:
                        for beta in lg.all_irrps():
                            u = np.zeros_like(v)
                            for op in ops:
                                u += np.conj(lg.character(beta, op)) * apply_operator(
                                    m, action_point_group(m, op), states, s2i, v)
                            u *= lg.irrep_dim(beta) / len(ops)
                            worst = max(worst, np.linalg.norm(
                                u - (v if beta == alpha else 0.0)))
                # spin flip
                overlap = np.vdot(v, apply_operator(
                    m, action_flip(m), states, s2i, v))
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
        sectors = m.build_sectors()
        for k in m.tr_group.all_momenta():
            lg, _ = m.little_group(k)
            ops = list(lg)
            basis, columns = kernel.momentum_sector(m, k)
            if columns.columns() == 0:
                continue
            traces = {}
            for op in ops:
                action = kernel.monomial_of(m, m._action_for(op))
                perm = _edkern.sector_action(columns, basis, action)
                count = len(perm.act)
                traces[tuple(np.asarray(op, dtype=float).ravel().round(8))] = float(np.real(
                    np.sum([perm.phase[x] for x in range(count) if perm.act[x] == x])))
            for alpha in (['A'] if len(ops) == 1 else lg.all_irrps()):
                d = 1 if len(ops) == 1 else lg.irrep_dim(alpha)
                expected = 0.0
                for op in ops:
                    chi = 1.0 if len(ops) == 1 else lg.character(alpha, op)
                    key = tuple(np.asarray(op, dtype=float).ravel().round(8))
                    expected += np.conj(chi) * traces[key]
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
# 4. multi-site cells, species and non-symmorphic elements
# ══════════════════════════════════════════════════════════════════════════════

def test_multi_site_cells(verbose: bool = True) -> bool:
    '''
    Species, sublattice exchange and an affine (glide) element.

    A symmetry operation may exchange sites of the *same species*: the two
    sublattices of a honeycomb are exchanged by the sixfold rotation.  It may
    not exchange sites of different species, and it may carry a fractional
    translation, which is what a glide plane is.
    '''
    ok = True
    a1 = np.array([1., 0.])
    a2 = np.array([0.5, np.sqrt(3.) / 2.])
    bond = (a1 + a2) / 3.
    centre = (a1 + a2) / 2.          # a hexagon centre of the honeycomb

    # (i) same species: the sixfold rotation is a symmetry and swaps the sublattices
    cell = {('C', 0): -bond / 2. - centre, ('C', 1): +bond / 2. - centre}
    model = Model(2, [(a1, 6), (a2, 6)], cell, point_group=twod.D6())
    swap = len(model.hilbert) == len(list(twod.D6()))
    # sites alternate between the two sublattices, so a swap sends even to odd
    swaps = int(model.actions[1].site_map(model.lattice)[0]) % 2 == 1
    ok &= swap and swaps
    if verbose:
        print(f"  {'✓' if swap and swaps else 'FAIL'}  honeycomb, one species: "
              f"{len(model.hilbert)}/{len(list(twod.D6()))} distinct permutations, "
              f"the 60° rotation exchanges the sublattices {swaps}")

    # (ii) different species: exchanging them is not a symmetry
    cell = {('C', 0): -bond / 2. - centre, ('O', 1): +bond / 2. - centre}
    rejected = False
    try:
        Model(2, [(a1, 6), (a2, 6)], cell, point_group=twod.D6())
    except ValueError:
        rejected = True
    ok &= rejected
    if verbose:
        print(f"  {'✓' if rejected else 'FAIL'}  honeycomb with two species: "
              f"the sublattice exchange is refused")

    # (iii) a glide: reflection in x followed by half a cell along y
    glide = AffineAction(np.array([[-1., 0.], [0., 1.]]), np.array([0., 0.5]))
    square = glide @ glide
    is_translation = bool(np.allclose(square.rotation, np.eye(2))
                          and np.allclose(square.translation, [0., 1.]))
    # on a torus one cell tall the square of the glide is the identity, so the
    # two elements are already a group
    cell = {('C', 0): np.array([0., 0.]), ('C', 1): np.array([0., 0.5])}
    model = Model(2, [(np.array([1., 0.]), 4), (np.array([0., 1.]), 1)], cell,
                  point_group=[np.eye(2), glide])
    accepted = (len(model.actions) == 2
                and np.array_equal((model.hilbert[1].site * model.hilbert[1].site).perm,
                                   np.arange(model.lattice.size_of_unit_cell)))
    ok &= is_translation and accepted
    if verbose:
        print(f"  {'✓' if is_translation and accepted else 'FAIL'}  glide on a two-site "
              f"cell: glide^2 is the translation {square.translation.tolist()} "
              f"{is_translation}, and the element is accepted {accepted}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 5. orbitals: a non-trivial local action
# ══════════════════════════════════════════════════════════════════════════════

def build_two_orbital_model() -> Model:
    '''
    Heisenberg on a 2x2 torus, with two p-like orbitals on every site.

    The local space is ``orbital ⊗ spin`` of dimension four and the orbital
    pair transforms as a circular doublet ``(x ± iy)``, so a rotation gives a
    phase and a reflection swaps the pair -- monomial in both cases.  The
    Hamiltonian is the identity on the orbital factor, hence invariant under
    ``P ⊗ R``.
    '''
    spin = np.eye(2, dtype=complex)
    pauli = [np.array([[0., 1.], [1., 0.]], dtype=complex),
             np.array([[0., -1j], [1j, 0.]], dtype=complex),
             np.array([[1., 0.], [0., -1.]], dtype=complex)]
    local_ops = [np.kron(spin, p) for p in pauli]
    model = Model(4, [(A1, 2), (A2, 2)], {('C', 0): np.array([0., 0.])},
                  point_group=twod.D4(),
                  orbital_mixing={'C': lambda rot: np.kron(circle_doublet(1)(rot), spin)},
                  local_charge=(0, 1, 0, 1))
    for i in range(2):
        for j in range(2):
            site = i * 2 + j
            right = i * 2 + (j + 1) % 2
            down = ((i + 1) % 2) * 2 + j
            # a two-site ring has one bond per direction, so the same bond
            # enters twice -- the model stays translation invariant
            model.add_coupling(Coupling(4, (site, right), (local_ops[0], local_ops[0]),
                                        (local_ops[1], local_ops[1]),
                                        (local_ops[2], local_ops[2])))
            model.add_coupling(Coupling(4, (site, down), (local_ops[0], local_ops[0]),
                                        (local_ops[1], local_ops[1]),
                                        (local_ops[2], local_ops[2])))
    return model


def test_orbital_mixing(tol: float = 1e-9, verbose: bool = True) -> bool:
    '''
    A non-trivial local action: spectra, sector labels and the group law.

    The local action is a phase per orbital, so the only thing that can go
    wrong is the phase itself -- which is exactly what the sector labels and
    the group-law check test.
    '''
    model = build_two_orbital_model()
    states = reference.basis_states(model)
    s2i = {state: row for row, state in enumerate(states)}
    direct = np.sort(reference.spectrum(reference.hamiltonian(model)))
    sectors = model.build_sectors()
    ev = np.sort(np.concatenate(
        [np.linalg.eigvalsh(sec.hamiltonian.toarray()) for sec in sectors.values()]))
    dim = sum(sec.dim for sec in sectors.values())
    spectra = (len(ev) == len(direct) == dim
               and np.max(np.abs(ev - direct)) < tol)

    # sector labels, with the full action (site permutation times local phase)
    worst = 0.0
    for key, sector in sectors.items():
        k, alpha, _ = key
        lg, _ = model.little_group(k)
        for column in range(sector.dim):
            v = np.asarray(sector.basis[:, column].todense()).ravel()
            for op in list(lg):
                if lg.irrep_dim(alpha) != 1:
                    continue
                overlap = np.vdot(v, apply_operator(
                    model, action_point_group(model, op), states, s2i, v))
                worst = max(worst, abs(overlap - lg.character(alpha, op)))
    labels = worst < tol

    # group law: U(a) U(b) = U(ab) on the computational basis
    pairs_ok = True
    probe = [states[0], states[len(states) // 3], states[-1]]
    for a, b in zip(model.actions, model.actions):
        composed = a @ b
        ua, ub = model._action_for(a.rotation), model._action_for(b.rotation)
        uab = model._action_for(composed.rotation)
        for state in probe:
            after_b, phase_b = ub.on_state(state)
            twice, phase_a = ua.on_state(after_b)
            once, phase_once = uab.on_state(state)
            pairs_ok &= (twice == once and abs(phase_a * phase_b - phase_once) < 1e-10)
    ok = spectra and labels and pairs_ok
    if verbose:
        print(f"  {'✓' if ok else 'FAIL'}  two p-like orbitals, D4: {len(sectors)} sectors, "
              f"Σ dim = {dim}/{len(direct)}, max |ΔE| = {np.max(np.abs(ev - direct)):.1e}, "
              f"labels {worst:.1e}, group law {pairs_ok}")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if _edkern is None:
        print('  skipped: the C++ kernel is not built (python cpp/build.py)')
        sys.exit(0)
    results = {}
    for name, fn in [('little groups', test_little_groups),
                     ('character tables', test_character_tables),
                     ('bad groups refused', test_rejects_bad_groups),
                     ('sector spectra', test_sector_spectra),
                     ('sector dimensions', test_dimensions_from_characters),
                     ('sector labels', test_sector_labels),
                     ('multi-site cells', test_multi_site_cells),
                     ('orbital mixing', test_orbital_mixing)]:
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
