from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.capital_allocation_phase4 import CapitalAllocationPhase4Comparison


DEFAULT_PROFILES = [
    "rookie_dealer_02_v2_66_ml_ranked",
    "rookie_dealer_02_v2_68_ml_ranked_exit_ai_050",
    "rookie_dealer_02_v2_70_ml_ranked_exit_ai_060",
    "rookie_dealer_02_v2_71_ml_ranked_exit_ai_050_scaled_buy",
    "rookie_dealer_02_v2_72_ml_ranked_exit_ai_scaled_buy_v2",
    "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue",
]
V2_73_PROFILE = "rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue"


@dataclass(frozen=True)
class CapitalAllocationPhase5Paths:
    markdown: Path
    json: Path
    purchase_audit_summary_md: Path


class CapitalAllocationPhase5Comparison(CapitalAllocationPhase4Comparison):
    def __init__(
        self,
        root: str | Path = ".",
        profiles: list[str] | None = None,
        start_date: str = "2023-01-01",
        end_date: str = "2026-05-31",
        focus_profile: str = V2_73_PROFILE,
    ) -> None:
        super().__init__(
            root=root,
            profiles=profiles or list(DEFAULT_PROFILES),
            start_date=start_date,
            end_date=end_date,
            focus_profile=focus_profile,
        )

    def save(self, result: dict[str, Any]) -> CapitalAllocationPhase5Paths:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        stem = "capital_allocation_phase5_v2_73_comparison_2023-01_to_2026-05"
        markdown = self.report_dir / f"{stem}.md"
        json_path = self.report_dir / f"{stem}.json"
        audit_md = self.report_dir / "v2_73_purchase_audit_summary_2023-01_to_2026-05.md"
        markdown.write_text(self.format_markdown(result).replace("Phase 4 v2_72", "Phase 5 v2_73"), encoding="utf-8")
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        audit_md.write_text(
            self.format_purchase_audit_markdown(result).replace("v2_72 Purchase Audit Summary", "v2_73 Purchase Audit Summary"),
            encoding="utf-8",
        )
        return CapitalAllocationPhase5Paths(markdown=markdown, json=json_path, purchase_audit_summary_md=audit_md)
