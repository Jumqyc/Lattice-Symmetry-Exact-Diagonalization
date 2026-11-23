'''
Independent references for the tests.

The library keeps no second implementation of the symmetry machinery, so the
tests have to bring their own oracle.  What they compare against is deliberately
the cheapest thing that is still independent: the computational basis written out
with ``itertools``, the Hamiltonian assembled densely straight from the couplings,
and dense diagonalisation.  Dense matrices are fine here, and only here.

Import this module from a test script, which runs with ``test/`` on the path.
'''

from __future__ import annotations

from itertools import product
from typing import Optional

import numpy as np


def basis_states(model, sztot: Optional[int] = None) -> list[tuple[int, ...]]:
    '''
    The computational basis of a charge sector, written out by hand.

    Args:
        model: the model.
        sztot: total charge, or None for the whole space.
    Returns:
        The digit tuples, in lexicographic order.  This is the definition the
        kernel's ``Basis`` has to reproduce, so it is written with ``itertools``
        and nothing else.
    '''
    dim = model.physical_dim
    charge = model.local_charge
    return [state for state in product(range(dim), repeat=model.size_of_unit_cell)
            if sztot is None or sum(charge[digit] for digit in state) == sztot]


def hamiltonian(model, sztot: Optional[int] = None) -> np.ndarray:
    '''
    The Hamiltonian of a charge sector as a dense matrix.

    Args:
        model: the model.
        sztot: total charge, or None for the whole space.
    Returns:
        A dense ``(n, n)`` complex matrix, assembled from the couplings alone:
        no symmetry, no kernel, no symmetrisation.  The couplings used in the
        tests respect the group, so this is the same operator the deck
        decomposition starts from.
    '''
    states = basis_states(model, sztot)
    index = {state: row for row, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)), dtype=complex)
    for row, state in enumerate(states):
        for coupling in model.couplings:
            for image, coefficient in coupling.coupling_pair(state).items():
                column = index.get(image)
                if column is not None:
                    matrix[row, column] += coefficient
    return matrix


def spectrum(matrix: np.ndarray) -> np.ndarray:
    '''
    The eigenvalues of a Hermitian matrix, ascending.

    Args:
        matrix: a dense Hermitian matrix.
    Returns:
        The eigenvalues, sorted.
    '''
    return np.sort(np.linalg.eigvalsh(matrix))


def sector_spectrum(sectors: dict) -> np.ndarray:
    '''
    The eigenvalues of the whole decomposition, ascending.

    Args:
        sectors: the dict returned by ``Model.build_sectors``.
    Returns:
        The union of the block spectra, sorted.
    '''
    return np.sort(np.concatenate([np.linalg.eigvalsh(sector.hamiltonian.toarray())
                                   for sector in sectors.values()]))
