'''A fingerprint of the spectra, for comparing before and after an optimisation.

The test suite checks the decomposition against independent oracles, which is what
makes it trustworthy and what makes it slow.  When the only question is "did this
change the numbers", a fingerprint answers it in seconds: per sector the size of
the block, its non-zeros, its trace and Frobenius norm, the lowest levels from
ARPACK, and the lowest levels from this package's own recurrence.

Usage:
    python bench/state.py --write     # record the current state
    python bench/state.py --check     # compare against the recorded state
'''

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'test'))

from test_csector import A1, A2, T1, T2, heisenberg  # noqa: E402
from test_pointgroup import heisenberg as rectangle  # noqa: E402

STATE = os.path.join(HERE, 'state.json')
LEVELS = 4
TOLERANCE = 1e-9


def models():
    '''The models the fingerprint covers, with their charge sector and spin flip.'''
    twod = importlib.import_module('geometry.2dsym')
    return [
        ('3x3 square, D4', heisenberg(3, twod.D4()), None, False),
        ('3x3 triangular, D6', heisenberg(3, twod.D6(), T1, T2), None, True),
        ('4x4 square, D4, Sz=8', heisenberg(4, twod.D4()), 8, False),
        ('4x5 rectangle, D2, Sz=10',
         rectangle(4, 5, twod.dihedral_of_order(2), A1, A2), 10, False),
    ]


def fingerprint(model, sztot, spin_flip: bool) -> dict:
    '''
    Per sector: the block's shape and weight, and the lowest levels.

    Args:
        model: the model.
        sztot: charge sector, or None for the whole space.
        spin_flip: whether to resolve the spin-flip parity.
    Returns:
        The sectors keyed by ``"k|alpha|s"``, each with the shape of its block,
        its non-zeros, its trace and Frobenius norm, and the lowest levels from
        ARPACK and from this package's recurrence.
    '''
    sectors = model.build_sectors(sztot=sztot, spin_flip=spin_flip)
    ours = model.spectra(sztot=sztot, count=LEVELS, spin_flip=spin_flip)
    out = {}
    for key in sorted(sectors, key=str):
        block = sectors[key].hamiltonian
        size = block.shape[0]
        asked = min(LEVELS, size)
        if size <= 12:
            # ARPACK refuses a request that leaves it fewer than two directions to
            # work with, and a block of a few rows is cheap to diagonalize whole
            reference = np.sort(np.linalg.eigvalsh(block.toarray()))[:asked]
        else:
            reference = np.sort(spla.eigsh(block, k=asked, which='SA', tol=0)[0])
        out['|'.join(str(part) for part in key)] = {
            'dim': int(size),
            'basis_nnz': int(sectors[key].basis.nnz),
            'block_nnz': int(block.nnz),
            'trace': float(block.diagonal().sum().real),
            'norm': float(np.sqrt((np.abs(block.data) ** 2).sum())),
            'arpack': [float(value) for value in reference],
            'ours': [float(value) for value in ours[key]],
        }
    return out


def compare(label: str, before: dict, now: dict) -> int:
    '''
    Compare two fingerprints of one model.

    Args:
        label: the model's name, for the messages.
        before: the recorded fingerprint.
        now: the fresh one.
    Returns:
        The number of differences found.
    '''
    failures = 0
    for key, entry in now.items():
        recorded = before.get(key)
        if recorded is None:
            print(f'   FAIL {label} {key}: not in the recorded state')
            failures += 1
            continue
        for field in ('dim', 'basis_nnz', 'block_nnz'):
            if recorded[field] != entry[field]:
                print(f'   FAIL {label} {key}.{field}: '
                      f'{recorded[field]} -> {entry[field]}')
                failures += 1
        for field in ('trace', 'norm'):
            if abs(recorded[field] - entry[field]) > TOLERANCE:
                print(f'   FAIL {label} {key}.{field}: '
                      f'{recorded[field]:.12g} -> {entry[field]:.12g}')
                failures += 1
        for field in ('arpack', 'ours'):
            if len(recorded[field]) != len(entry[field]):
                print(f'   FAIL {label} {key}.{field}: wrong length')
                failures += 1
                continue
            for index, value in enumerate(entry[field]):
                if abs(recorded[field][index] - value) > TOLERANCE:
                    print(f'   FAIL {label} {key}.{field}[{index}]: '
                          f'{recorded[field][index]:.12f} -> {value:.12f}')
                    failures += 1
    for key in before:
        if key not in now:
            print(f'   FAIL {label} {key}: recorded but not produced')
            failures += 1
    return failures


def main() -> int:
    '''Record or check the fingerprint of every model.'''
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true', help='record the state')
    mode.add_argument('--check', action='store_true', help='compare with it')
    arguments = parser.parse_args()

    recorded = {}
    if arguments.check:
        if not os.path.exists(STATE):
            print(f'no recorded state at {STATE}; run --write first')
            return 2
        recorded = json.load(open(STATE))
    failures = 0
    total = 0.0
    fresh = {}
    for label, model, sztot, spin_flip in models():
        start = time.perf_counter()
        print(f'== {label}', flush=True)
        now = fingerprint(model, sztot, spin_flip)
        elapsed = time.perf_counter() - start
        total += elapsed
        print(f'   {len(now)} sectors in {elapsed:.2f} s', flush=True)
        fresh[label] = now
        if arguments.check:
            failures += compare(label, recorded.get(label, {}), now)
    if arguments.check:
        print(f'== {failures} differences in {total:.2f} s')
        return 1 if failures else 0
    with open(STATE, 'w') as handle:
        json.dump(fresh, handle, indent=1, sort_keys=True)
    print(f'== wrote {STATE} ({os.path.getsize(STATE)} bytes) in {total:.2f} s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
