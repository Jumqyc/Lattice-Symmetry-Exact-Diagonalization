'''
Build the C++ kernel into an importable extension module.

The module is written next to the Python sources as ``_edkern<ext>`` so that
``import _edkern`` works from the repository root, and it is *not* committed: a
clone builds it with ``python cpp/build.py``.

Requires ``pybind11`` (headers only) and Eigen (``/usr/include/eigen3`` on
Debian-like systems).  No BLAS or LAPACK development files are needed.
'''

import os
import pathlib
import subprocess
import sys
import sysconfig

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TARGET = ROOT / ('_edkern' + sysconfig.get_config_var('EXT_SUFFIX'))


def main(extra_flags=()) -> None:
    '''
    Compile ``cpp/*.cpp`` into the extension module.

    Args:
        extra_flags: extra compiler flags, e.g. ``('-march=native',)``; the
            defaults are portable.
    Raises:
        SystemExit: if pybind11 is missing.
        subprocess.CalledProcessError: if the compiler fails.
    '''
    try:
        import pybind11
    except ImportError:
        raise SystemExit('pybind11 is required to build the kernel: '
                         'uv pip install pybind11')

    sources = sorted(str(p) for p in HERE.glob('*.cpp'))
    if not sources:
        raise SystemExit(f'no C++ sources in {HERE}')

    command = [
        os.environ.get('CXX', 'g++'),
        '-O3', '-std=c++17', '-shared', '-fPIC', '-Wall',
        '-o', str(TARGET),
        f'-I{HERE}',
        # third-party headers are system headers: their own warnings are noise
        f'-isystem{pybind11.get_include()}',
        f'-isystem{sysconfig.get_paths()["include"]}',
        f'-isystem{os.environ.get("EIGEN_INCLUDE", "/usr/include/eigen3")}',
        *extra_flags,
        *sources,
    ]
    print(' '.join(command))
    subprocess.run(command, check=True)
    print(f'built {TARGET}')


if __name__ == '__main__':
    main(extra_flags=('-march=native',) if '--native' in sys.argv else ())
