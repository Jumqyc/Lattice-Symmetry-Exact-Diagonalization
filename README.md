# LattSymED

Exact diagonalisation of lattice models resolved by the symmetry of the lattice:
translation sectors (momentum labels), the little group at each momentum, induced
representations of the point group, monomial orbital actions, and the spin flip.

The group theory, the model description and the API are Python.  Everything that
loops over states, columns or group elements is a C++ extension, and the Python
side builds nothing but the arrays the kernel takes: the sectors are produced by
the kernel and handed back as sparse matrices.

Every object that scales with the Hilbert space is sparse -- the computational
basis, the momentum sectors, the adapted vectors, the blocks and the Lanczos
recurrence itself -- and no dense matrix of the size of a sector is formed
anywhere in the library.  Dense matrices appear only in the tests, where they are
the oracle the sparse results are compared against.

## Layout

| path | contents |
| --- | --- |
| `base.py` | the abstract `PointGroup`, `LatticeAction`, `HilbertAction` |
| `representation.py` | finite point groups, character tables, cosets |
| `lattice_action.py` | `Lattice`, the translation group, affine (space-group) actions |
| `hilbert_action.py` | how an action acts on the computational basis |
| `geometry/2dsym.py` | `C_n` and `D_n` in two dimensions |
| `ed.py` | `Model`, `Coupling`, sectors and their spectra |
| `kernel.py` | the bridge: the model as the arrays the kernel takes, and back |
| `cpp/` | the kernel: basis, monomial actions, Hamiltonian, sectors, solvers |
| `test/` | one script per layer, each checked against an independent reference |
| `doc/` | the write-up |
| `demo.ipynb` | a tour of the lattice layer |

## Setup

```sh
uv venv .venv
uv pip install -r requirements.txt
.venv/bin/python cpp/build.py
```

`Model.build_sectors` runs on the kernel through `kernel.py`.  The tests take
their oracle from `test/reference.py`, which writes the computational basis out
with `itertools` and assembles the Hamiltonian densely straight from the
couplings: dense matrices live there and nowhere else.

`cpp/build.py` compiles `cpp/*.cpp` into `_edkern<ext>` in the repository root;
that file is a build artefact and is not committed.  It needs a C++17 compiler
and nothing else: the kernel's linear algebra is sparse, and the one dense solve
it does not do.  Add `--native` to build with `-march=native`.

## Tests

Each file is a standalone script, not a pytest module: the checks take a model as
an argument, so pytest would collect them as broken tests.  Run them from the
repository root:

```sh
.venv/bin/python test/test_heisenberg.py     # momentum sectors against direct ED
.venv/bin/python test/test_pointgroup.py     # little groups, sectors, labels, orbitals
.venv/bin/python test/test_induced_char.py   # induced characters, one momentum per star
.venv/bin/python test/test_cgroup.py         # kernel character tables vs Python
.venv/bin/python test/test_chilbert.py       # kernel basis, actions, momentum basis
.venv/bin/python test/test_csector.py        # kernel Hamiltonian, sectors, Lanczos
```

`test_cgroup.py`, `test_chilbert.py` and `test_csector.py` skip themselves when
the kernel is not built.
