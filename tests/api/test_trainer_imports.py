"""The training entrypoint imports in an environment without IPython.

The Modal image installs the lock without development tools, so IPython is absent there.
The trainer reaches ``fastprogress`` through opifex, artifex and blackjax, and a
``fastprogress`` release that imports IPython at module level breaks every training run
before it starts. The child blocks IPython the way Python documents for an absent
package (``sys.modules[name] = None``) rather than relying on what this interpreter has.
"""

from __future__ import annotations

from substrax.testing import run_python


_IMPORT_TIMEOUT_SECONDS = 300.0

_PROGRAM = """
import sys

sys.modules["IPython"] = None
import diffav.api.map_conditioned_trainer  # noqa: F401
"""


def test_map_conditioned_trainer_imports_without_ipython() -> None:
    """Importing the trainer must not require IPython."""
    result = run_python(_PROGRAM, timeout=_IMPORT_TIMEOUT_SECONDS)
    assert result.returncode == 0, result.stderr
