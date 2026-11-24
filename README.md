# LattSymED

Exact diagonalisation of lattice models resolved by the symmetry of the lattice:
translation sectors (momentum labels), the little group at each momentum, induced
representations of the point group, monomial orbital actions, and the spin flip.

The group theory, the model description and the API are Python.  Everything that
loops over states, columns or group elements is a C++ extension, and the Python
side builds nothing but the arrays the kernel takes: the sectors are produced by
the kernel and handed back as sparse matrices.

Every object that scales with the Hilbert space is sparse, and the sector is
carried as one thing: `SectorSpec` holds a sector's key, its dimension and the
induced character of the whole lattice group, together with the momentum and
little-group characters that character is built from.  The projection reads only
that.  Nothing of the size of the basis is written down except the state list and
the orbit labels; the block of a sector is either built once (`cache=True`) or
never built (`cache=False`).  Dense matrices appear only in the tests, where they
are the oracle the sparse results are compared against.

## Layout

| path | contents |
| --- | --- |
| `base.py` | the abstract `PointGroup`, `LatticeAction`, `HilbertAction` |
| `representation.py` | finite point groups, character tables, cosets |
| `lattice_action.py` | `Lattice`, the translation group, affine (space-group) actions |
| `hilbert_action.py` | how an action acts on the computational basis |
| `geometry/2dsym.py` | `C_n` and `D_n` in two dimensions |
| `ed.py` | `Model`, `Coupling`, `SectorSpec`, sectors and their spectra |
| `kernel.py` | the bridge: the model as the arrays the kernel takes, and back |
| `cpp/` | the kernel: basis, monomial actions, Hamiltonian, sectors, solvers |
| `test/` | one script per layer, each checked against an independent reference |
| `doc/` | the write-up |
| `demo.ipynb` | a tour of the lattice layer |

## What it computes

```python
model.spectra(sztot=8, count=4)                 # build each sector's block, then solve it
model.spectra(sztot=8, count=4, cache=False)    # never build it: apply the product per step
model.build_sectors(sztot=8)                    # keep the adapted vectors and the blocks
for spec in model.sector_specs(sztot=8):        # one sector with the whole group's character
    ...
```

A sector is a pair (momentum, little-group irrep); `SectorSpec` carries its induced
character on every element of the lattice group together with the two factors the
projection applies -- the momentum character on the translations and the little
group's on its own elements.  At 4x4 half filling the two modes above take 0.92 s and
12.35 s, and the block mode is what a lattice that fits should use.

## Setup

```sh
uv venv .venv
uv pip install -r requirements.txt
.venv/bin/python cpp/build.py
```

`Model.build_sectors` runs on the kernel through `kernel.py`.  The momentum
sectors are independent, so they are spread over threads (`kernel.workers`, one
per core by default, capped by `kernel.memory_budget` because each thread holds a
scratch of the size of the basis); the kernel releases the interpreter lock while
it walks the Hilbert space, so these are real threads and not a formality.  The tests take
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
.venv/bin/python test/test_chilbert.py       # kernel basis, actions, momentum basis
.venv/bin/python test/test_csector.py        # sector blocks, their spectra, Lanczos
.venv/bin/python test/test_spin_one.py       # AKLT: spin one, a bulk singlet, edge states
```

`test_chilbert.py` and `test_csector.py` skip themselves when the kernel is not
built.  `bench/state.py --check` compares a fingerprint of four models (per sector
the shape, non-zeros, trace and norm of the block, and the lowest levels) in about
twelve seconds, which is what to run while changing the kernels.
