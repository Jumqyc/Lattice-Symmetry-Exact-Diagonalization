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


class Permutation:
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
