from pathlib import Path

import pytest

from stategate import Policy

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def refund_policy() -> Policy:
    return Policy.from_file(EXAMPLES / "refund_policy.yaml")


@pytest.fixture
def policy_path() -> Path:
    return EXAMPLES / "refund_policy.yaml"
