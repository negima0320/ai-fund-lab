from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from main import load_config


@pytest.fixture
def config() -> dict:
    return load_config(ROOT / "config" / "rookie_dealer.yaml")


@pytest.fixture
def config_copy(config: dict) -> dict:
    return deepcopy(config)
