'''
How a symmetry acts on the computational basis.

The geometric side of an operation (``lattice_action.py``) says where the sites
and the momentum labels go.  This module says what the operation does to a
basis state of the Hilbert space.

The interface is deliberately narrow: an operation may only map one basis state
onto *one* basis state, times a phase,

    U(g)|σ> = φ(σ) |σ'>.

That is the property that keeps every projection in the computational basis
cheap.  An operation whose local action mixes components (a genuine rotation of
two orbitals, say) is not expressible here, and it must not be: a mixing
operation turns one basis state into an exponential superposition, which is
exactly what the computational-basis construction exists to avoid.  Any
monomial local basis avoids it -- in two dimensions the circular harmonics
(x ± iy)^m always do.

The phase carries the Bloch phase of the momentum basis and, later, the
monomial phase of a local orbital rotation.
'''

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from base import HilbertAction


class Permutation(HilbertAction):
    '''
    A site permutation of the unit cell.

    ``perm[i]`` is the site that is *mapped onto* site ``i``, i.e. the
    preimage of ``i``.  Acting on a computational basis state this gives

        ``perm(state)[i] == state[perm[i]] == state[g^{-1}(i)]``,

    which is exactly ``g|σ_1 … σ_L> = |σ_{g^{-1}(1)} … σ_{g^{-1}(L)}>``.
    With the preimage convention the product ``a * b`` is the group element
    ``a b`` (apply ``b`` first), so ``Permutation`` is a genuine, not an
    anti-, representation of the symmetry group on the computational basis.
    '''
    def __init__(self, 
                 perm: np.ndarray):
        self.perm:NDArray[np.int64] = np.asarray(perm, dtype=int)
        self.size = len(perm)
        if not np.array_equal(np.sort(perm), np.arange(self.size)):
            raise ValueError("Invalid permutation array. It must contain all integers from 0 to size-1 exactly once.")

    def __call__(self, x:np.ndarray|int) -> np.ndarray|int:
        '''Action on a basis state (``x[self.perm]``) or on a site index.'''
        if isinstance(x, int):
            return self.perm[x]
        elif isinstance(x, np.ndarray):
            return x[self.perm]

    def on_state(self, state: tuple[int, ...]) -> tuple[tuple[int, ...], complex]:
        '''``U(g)|state> = |state'>`` with phase one.'''
        return tuple(self(np.array(state))), 1.0 + 0.0j

    def __mul__(self, other: 'Permutation') -> 'Permutation':
        if not isinstance(other, Permutation):
            raise TypeError("Can only multiply with another Permutation.")
        if self.size != other.size:
            raise ValueError("Permutations must be of the same size to multiply.")
        return Permutation(self.perm[other.perm])
    def inv(self)->Permutation:
        inv_perm = np.argsort(self.perm)
        return Permutation(inv_perm)
    def __pow__(self, power: int) -> 'Permutation':
        if power == 0:
            return Permutation(np.arange(self.size))
        elif power > 0:
            result = self
            for _ in range(power - 1):
                result = result * self
            return result
        else:
            return self.inv() ** (-power)
    def __hash__(self):
        return hash(tuple(self.perm))


def from_lattice_action(action, lattice) -> Permutation:
    '''
    Turn a geometric operation into its action on the computational basis.

    The geometric action moves the sites; on the basis states it acts as a site
    permutation, ``U(g)|σ_1 … σ_L> = |σ_{g^{-1}(1)} … σ_{g^{-1}(L)}>``, which is
    what :class:`Permutation` stores.  Any local orbital action is monomial (the
    interface of this module), so it contributes a phase rather than a
    superposition; that phase is carried by the momentum basis, not here.

    Args:
        action: a ``LatticeAction``, i.e. something with ``site_map(lattice)``.
        lattice: the lattice the operation acts on.
    Returns:
        The site permutation of the operation.
    Raises:
        ValueError: if the operation does not permute the sites.
    '''
    image = action.site_map(lattice)
    return Permutation(np.argsort(image))


def extract_monomial(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Read a monomial unitary as a local permutation plus a phase per index.

    ``matrix`` is the local action of one group element on the local basis.  It
    must be a *monomial* unitary -- exactly one non-zero entry per column --
    because that is what makes it act on a basis state as

        U(g)|σ> = Π_i φ(σ_i) |σ'>,  σ'_i = π(σ_i).

    Args:
        matrix: ``(d, d)`` unitary acting on the local space.
    Returns:
        (perm, phase) with ``perm[i]`` the new local index of basis vector ``i``
        and ``phase[i]`` the coefficient.
    Raises:
        ValueError: if the matrix is not unitary, or not monomial.  A mixing
            local action would map one basis state onto an exponential
            superposition, which the computational-basis construction cannot
            afford; re-expressing the orbitals in a monomial basis (the
            circular harmonics ``(x ± iy)^m`` for two dimensions) fixes it.
    '''
    m = np.asarray(matrix, dtype=complex)
    if m.shape[0] != m.shape[1]:
        raise ValueError(f"The local action must be square, got {m.shape}.")
    if not np.allclose(m @ m.conj().T, np.eye(m.shape[0]), atol=1e-8):
        raise ValueError("The local action must be unitary.")
    perm = np.empty(m.shape[0], dtype=int)
    phase = np.empty(m.shape[0], dtype=complex)
    for i in range(m.shape[1]):
        column = m[:, i]
        nz = np.flatnonzero(np.abs(column) > 1e-8)
        if len(nz) != 1:
            raise ValueError(
                f"The local action is not monomial: column {i} has {len(nz)} non-zero "
                "entries.  Re-express the orbitals in a monomial basis -- in two "
                "dimensions the circular harmonics (x ± iy)^m always allow it.")
        perm[i] = nz[0]
        phase[i] = column[nz[0]]
    return perm, phase


def circle_doublet(m: int = 1):
    '''
    Local action of a pair of circular harmonics ``(x ± iy)^m``.

    A rotation by θ multiplies them by ``exp(±imθ)`` and a reflection maps one
    onto the other, so both are monomial: the pair never mixes.

    Args:
        m: the weight of the doublet; 1 for a p pair, 2 for a d pair.
    Returns:
        A function ``rotation -> (2, 2) complex matrix``.
    '''
    def local_action(rotation: np.ndarray) -> np.ndarray:
        rot = np.asarray(rotation, dtype=float)
        if np.linalg.det(rot) > 0:
            theta = np.arctan2(rot[1, 0], rot[0, 0])
            return np.diag([np.exp(1j * m * theta), np.exp(-1j * m * theta)])
        phi = np.arctan2(rot[1, 0], rot[0, 0]) / 2.0
        return np.array([[0.0, np.exp(-2j * m * phi)],
                         [np.exp(2j * m * phi), 0.0]], dtype=complex)
    return local_action


class MonomialAction(HilbertAction):
    '''
    One symmetry operation on the computational basis: a site permutation plus a
    monomial local action.

        U(g)|σ> = Π_i φ_{s_i}(σ_i) |σ'>,

    where the species of the source site selects the local permutation and
    phase, and the site permutation moves the index.

    Attributes:
        site: the site permutation (preimage convention, see ``Permutation``).
        species: species of every site, in site-index order.
        local: species -> ``(perm, phase)`` of the local action.
    '''

    def __init__(self, site: Permutation, species: list[str],
                 local: dict[str, tuple[np.ndarray, np.ndarray]]):
        self.site = site
        self.species = species
        self.local = local

    def on_state(self, state: tuple[int, ...]) -> tuple[tuple[int, ...], complex]:
        '''``U(g)|state> = phase |state'>``.'''
        source = self.site.perm
        new = [0] * len(state)
        phase = 1.0 + 0.0j
        for i in range(len(state)):
            src = source[i]
            value = state[src]
            action = self.local.get(self.species[src])
            if action is None:
                new[i] = value
            else:
                new[i] = int(action[0][value])
                phase *= action[1][value]
        return tuple(new), phase

    def after_translation(self, translation: Permutation) -> 'MonomialAction':
        '''
        Compose a translation in front of this action.

        Translations have a trivial local action, so only the site permutation
        changes.

        Args:
            translation: the site permutation of the translation.
        Returns:
            The composed action.
        '''
        return MonomialAction(translation * self.site, self.species, self.local)
