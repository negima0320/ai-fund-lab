from __future__ import annotations

from pathlib import Path

from ml.position_sizing_phase2 import PositionSizingPhase2SoftRules
from tests.test_ml_position_sizing_phase1 import PROFILE
from tests.test_ml_position_sizing_phase1 import _write_fake_backtest


def test_position_sizing_phase2_soft_rules(tmp_path: Path) -> None:
    _write_fake_backtest(tmp_path)
    simulation = PositionSizingPhase2SoftRules(root=tmp_path, profiles=[PROFILE])

    result = simulation.build()
    paths = simulation.save(result)

    rows = {(row["profile"], row["sizing_rule"]): row for row in result["summary"]}
    assert rows[(PROFILE, "baseline")]["adjusted_net_profit"] == 250.0
    assert rows[(PROFILE, "bad_entry_defensive_soft")]["adjusted_net_profit"] == 285.0
    assert rows[(PROFILE, "bad_entry_defensive_very_soft")]["adjusted_net_profit"] == 270.0
    assert rows[(PROFILE, "expected_return_soft")]["adjusted_net_profit"] == 320.0
    assert rows[(PROFILE, "combined_soft")]["adjusted_net_profit"] == 280.0
    assert paths.markdown.exists()
    assert paths.json.exists()
    assert paths.summary_csv.exists()
