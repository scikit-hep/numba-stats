import importlib
import pkgutil
import subprocess as subp
import sys

import pytest

import numba_stats

all_modules = []
for _, modname, ispkg in pkgutil.walk_packages(numba_stats.__path__):
    if modname.startswith("_"):
        continue
    all_modules.append(modname)


@pytest.mark.parametrize("module", all_modules)
def test_all(module):
    pytest.importorskip("pydocstyle")
    m = importlib.import_module(f"numba_stats.{module}")
    r = subp.run([sys.executable, "-m", "pydocstyle", m.__file__], stdout=subp.PIPE)
    rc = int(r.returncode)
    assert rc == 0, r.stdout.decode("utf8")
