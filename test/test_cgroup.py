'''
Test of the C++ group kernel against the Python character tables.

The kernel is given nothing but a multiplication table; it has to recover the
conjugacy classes, the irrep dimensions and the characters on its own.  Three
things are checked for every group:

* the row set of the character table equals the Python table (the labels may
  differ, the characters may not);
* the great orthogonality theorem holds for the table the kernel returns;
* the dimensions square to the group order.

Groups tested: the named point groups, and the *little groups* that actually
occur in a lattice -- which are generic subgroups, i.e. exactly the case the
kernel has to get right.

Requires the kernel: build it with ``python cpp/build.py``.
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
import numpy as np

twod = importlib.import_module('geometry.2dsym')
from ed import Model
from lattice_action import label_action
from representation import _op_key, ExplicitPointGroup

try:
    import _edkern
except ImportError:                                  # pragma: no cover
    _edkern = None


def multiplication_table(ops: list[np.ndarray]) -> list[list[int]]:
    '''
    Multiplication table of a group given by its operation matrices.

    Args:
        ops: the operations, without duplicates.
    Returns:
        ``table[a][b]`` is the index of the product ``ops[a] @ ops[b]``.
    '''
    index = {_op_key(op): i for i, op in enumerate(ops)}
    return [[index[_op_key(ops[a] @ ops[b])] for b in range(len(ops))]
            for a in range(len(ops))]


def check_group(ops: list[np.ndarray], python_chars: np.ndarray, label: str,
                tol: float = 1e-8, verbose: bool = True) -> bool:
    '''
    Compare one kernel table with the Python table of the same group.

    Args:
        ops: the operations of the group, without duplicates.
        python_chars: ``(n_irreps, n_elements)`` Python characters.
        label: name used in the log.
        tol: absolute tolerance.
        verbose: print the outcome.
    Returns:
        True if every check passed.
    '''
    n = len(ops)
    table = _edkern.character_table(multiplication_table(ops))
    dims = np.asarray(table.dims)
    chars = np.asarray(table.chars)                 # (n_irreps, n_classes)
    expanded = chars[:, np.asarray(table.class_of)]  # (n_irreps, n_elements)
    sizes = np.asarray([len(c) for c in table.classes])

    same = (len(dims) == len(python_chars)
            and all(any(np.allclose(row, other, atol=tol) for other in python_chars)
                    for row in expanded))
    gram = (chars * sizes[None, :]) @ chars.conj().T
    orthogonal = np.allclose(gram, n * np.eye(len(dims)), atol=tol)
    complete = abs(np.sum(dims ** 2) - n) < tol
    ok = same and orthogonal and complete
    if verbose:
        print(f"  {'✓' if ok else 'FAIL'}  {label}: |G| = {n}, {len(dims)} irreps, "
              f"dims {np.round(dims, 2).tolist()}, matches Python {same}, "
              f"orthogonal {orthogonal}, Σd² = |G| {complete}")
    return ok


def named_groups(verbose: bool = True) -> bool:
    '''The named point groups against their hand-written tables.'''
    ok = True
    for pg in [twod.C2(), twod.C3(), twod.C4(), twod.C6(),
               twod.D3(), twod.D4(), twod.D6()]:
        ops = list(pg)
        named = np.array([[pg.character(a, op) for op in ops] for a in pg.all_irrps()])
        ok &= check_group(ops, named, type(pg).__name__, verbose=verbose)
    return ok


def little_groups(verbose: bool = True) -> bool:
    '''Every little group of a few lattices against its generic Python table.'''
    a1 = np.array([1., 0.])
    a2 = np.array([0., 1.])
    t1 = np.array([1., 0.])
    t2 = np.array([0.5, np.sqrt(3.) / 2.])
    cases = [('4x4 square, D4', 4, 4, twod.D4(), a1, a2),
             ('3x3 triangular, D6', 3, 3, twod.D6(), t1, t2),
             ('2x4 rectangle, C2', 2, 4, twod.C2(), a1, a2)]
    ok = True
    for label, n1, n2, pg, b1, b2 in cases:
        model = Model(2, [(b1, n1), (b2, n2)], {('A', 0): np.array([0., 0.])},
                      point_group=pg)
        seen = {}
        for k in model.tr_group.all_momenta():
            lg, _ = model.little_group(k)
            ops = [op for op in list(lg)]
            if len(ops) == 1:
                continue        # the trivial group has nothing to compare
            key = tuple(sorted(_op_key(op) for op in ops))
            if key in seen:
                continue
            python_chars = np.array([[lg.character(a, op) for op in ops]
                                     for a in lg.all_irrps()])
            seen[key] = check_group(ops, python_chars,
                                    f'{label} little group of order {len(ops)}',
                                    verbose=verbose)
        ok &= all(seen.values())
    return ok


def main():
    if _edkern is None:
        print('  skipped: the C++ kernel is not built (python cpp/build.py)')
        sys.exit(0)
    results = {}
    for name, fn in [('named groups', named_groups),
                     ('little groups', little_groups)]:
        print(f'── {name} ──')
        results[name] = fn()
        print()
    failed = [k for k, v in results.items() if not v]
    if failed:
        print('FAILED: ' + ', '.join(failed))
        sys.exit(1)
    print('The C++ character tables agree with the Python ones.')


if __name__ == '__main__':
    main()
