from __future__ import annotations

import pytest

from icilval.spec import load_spec


@pytest.fixture(scope="session")
def spec():
    return load_spec()


@pytest.fixture(scope="session")
def track(spec):
    """The field a test acts on when it is not about fields.

    Most of the suite predates the second one: it exercises the store, unit derivation or the
    submission check, which work the same in any field. Naming one keeps them honest now that
    `Spec.sole_track` refuses to guess.
    """
    return "sensorimotor"
