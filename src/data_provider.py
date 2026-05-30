"""Data provider boundary for market data sources."""

from __future__ import annotations

import os
import random
import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request


SECTORS = [
    "情報・通信",
    "電気機器",
    "機械",
    "化学",
    "サービス",
    "小売",
    "医薬品",
    "輸送用機器",
    "銀行",
    "卸売",
]


class BaseDataProvider(ABC):
    """Common interface for stock data providers."""

    @abstractmethod
    def get_listed_stocks(self) -> list[dict[str, Any]]:
        """Return listed stock metadata."""

    @abstractmethod
    def get_daily_prices(self, date: Optional[date] = None) -> list[dict[str, Any]]:
        """Return daily price data."""

    @abstractmethod
    def get_stock_fundamentals(self, code: str) -> dict[str, Any]:
        """Return stock fundamentals."""

    @abstractmethod
    def get_news(self, code: str) -> list[dict[str, Any]]:
        """Return stock news."""


class DummyDataProvider(BaseDataProvider):
    """Deterministic local dummy data provider used by demo mode."""

    def __init__(self, config: dict[str, Any], run_date: date, run_id: str) -> None:
        self.config = config
        self.run_date = run_date
        self.run_id = run_id

    def get_listed_stocks(self) -> list[dict[str, Any]]:
        candidate_count = int(self.config["universe"]["candidate_count"])
        return [
            {
                "code": f"{1301 + index:04d}",
                "name": f"東証プライムDummy{index + 1:02d}",
                "market": self.config["universe"]["market"],
                "sector": SECTORS[index % len(SECTORS)],
            }
            for index in range(candidate_count)
        ]

    def get_daily_prices(self, date: Optional[date] = None) -> list[dict[str, Any]]:
        target_date = date or self.run_date
        rng = random.Random(f"{target_date.isoformat()}:{self.run_id}:prices")
        prices = []
        for stock in self.get_listed_stocks():
            momentum = round(rng.uniform(-0.04, 0.12), 4)
            volume_ratio = round(rng.uniform(0.8, 3.2), 2)
            volatility = round(rng.uniform(0.01, 0.08), 4)
            prices.append(
                {
                    "code": stock["code"],
                    "close_price": rng.randint(700, 6500),
                    "momentum_5d": momentum,
                    "volume_ratio_20d": volume_ratio,
                    "volatility_20d": volatility,
                }
            )
        return prices

    def get_stock_fundamentals(self, code: str) -> dict[str, Any]:
        rng = random.Random(f"{self.run_id}:{code}:fundamentals")
        return {
            "code": code,
            "per": round(rng.uniform(8.0, 35.0), 2),
            "pbr": round(rng.uniform(0.6, 5.0), 2),
            "roe": round(rng.uniform(0.02, 0.18), 4),
        }

    def get_news(self, code: str) -> list[dict[str, Any]]:
        return [
            {
                "code": code,
                "headline": "ダミーニュース: 短期材料を確認",
                "source": "dummy",
            }
        ]


class JQuantsDataProvider(BaseDataProvider):
    """J-Quants V2/API key provider skeleton.

    Real API fetches are intentionally left as TODO for the next phase.
    """

    def __init__(self, env_path: Optional[Path] = None, timeout_seconds: int = 20) -> None:
        _load_env(env_path)
        api_key = os.getenv("JQUANTS_API_KEY")
        if not api_key:
            raise RuntimeError("JQUANTS_API_KEY が未設定です")

        self.api_key = api_key
        self.base_url = "https://api.jquants.com/v2"
        self.default_headers = {"x-api-key": self.api_key}
        self.timeout_seconds = timeout_seconds

    def get_listed_stocks(self) -> list[dict[str, Any]]:
        payload = self._get_json("/equities/master")
        return _extract_records(payload)

    def get_daily_prices(self, date: Optional[date] = None) -> list[dict[str, Any]]:
        if date is None:
            raise ValueError("date is required for J-Quants daily prices fetch.")
        date_key = date.strftime("%Y%m%d")
        return self._get_paginated_records("/equities/bars/daily", {"date": date_key})

    def get_daily_prices_range(self, start_date: date, end_date: date) -> list[dict[str, Any]]:
        return self._get_paginated_records(
            "/equities/bars/daily",
            {
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
            },
        )

    def get_stock_fundamentals(self, code: str) -> dict[str, Any]:
        # TODO: Fetch fundamentals/statements from J-Quants V2 API with x-api-key.
        raise NotImplementedError("J-Quants fundamentals fetch is not implemented yet.")

    def get_news(self, code: str) -> list[dict[str, Any]]:
        # J-Quants does not provide news. Use another API or web search in a later phase.
        return []

    def _build_request(self, path: str) -> Request:
        """Build a future J-Quants V2 HTTP request with x-api-key authentication."""
        return Request(f"{self.base_url}{path}", headers=self.default_headers)

    def _get_json(self, path: str) -> Any:
        request = self._build_request(path)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise RuntimeError("J-Quants authentication failed. Check JQUANTS_API_KEY.") from exc
            if exc.code == 429:
                raise RuntimeError("J-Quants API rate limit exceeded. Wait a while and retry.") from exc
            raise RuntimeError(f"J-Quants API request failed with HTTP {exc.code}.") from exc
        except URLError as exc:
            raise RuntimeError(f"J-Quants network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("J-Quants network error: request timed out.") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("J-Quants response format is invalid JSON.") from exc

    def _get_paginated_records(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        pagination_key = ""
        while True:
            request_params = dict(params)
            if pagination_key:
                request_params["pagination_key"] = pagination_key
            payload = self._get_json(f"{path}?{urlencode(request_params)}")
            records.extend(_extract_records(payload))
            if not isinstance(payload, dict):
                break
            pagination_key = str(payload.get("pagination_key") or "")
            if not pagination_key:
                break
        return records


def _load_env(env_path: Optional[Path]) -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        _load_env_without_dotenv(env_path)
        return
    load_dotenv(env_path)


def _load_env_without_dotenv(env_path: Optional[Path]) -> None:
    if env_path is None or not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise RuntimeError("J-Quants response format is invalid: expected object or list.")

    for key in ("data", "master", "equities", "listed", "info"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    for value in payload.values():
        if isinstance(value, list):
            return value

    raise RuntimeError("J-Quants response format is invalid: listed stock records not found.")
