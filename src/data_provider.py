"""Data provider boundary for market data sources."""

from __future__ import annotations

import os
import random
import json
from datetime import date, datetime
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit
from urllib.request import Request

from earnings_calendar import normalize_earnings_calendar_records
from investor_context import normalize_investor_type_records
from jquants_plan import jquants_capabilities, jquants_has_capability, normalize_jquants_plan


JQUANTS_ENDPOINTS: dict[str, dict[str, str]] = {
    "listed_info": {"path": "/equities/master", "cache_dir": "listed_info", "filename": "YYYY-MM-DD.json"},
    "prices": {"path": "/equities/bars/daily", "cache_dir": "prices", "filename": "YYYY-MM-DD.json"},
    "financial_statements": {"path": "/fins/summary", "cache_dir": "financial_statements", "filename": "YYYY-MM-DD_to_YYYY-MM-DD.json"},
    "earnings_calendar": {"path": "/equities/earnings-calendar", "cache_dir": "earnings_calendar", "filename": "YYYY-MM-DD(.json)|YYYY-MM-DD_to_YYYY-MM-DD.json"},
    "trading_calendar": {"path": "/markets/calendar", "cache_dir": "trading_calendar", "filename": "YYYY-MM-DD_to_YYYY-MM-DD.json"},
    "investor_types": {"path": "/equities/investor-types", "cache_dir": "investor_types", "filename": "YYYY-MM-DD_to_YYYY-MM-DD.json"},
    "topix_prices": {"path": "/indices/bars/daily/topix", "cache_dir": "topix_prices", "filename": "YYYY-MM-DD_to_YYYY-MM-DD.json"},
}


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


class RateLimiter:
    """Simple thread-safe fixed-interval limiter for J-Quants API calls."""

    def __init__(self, requests_per_minute: int, sleeper: Any = time.sleep, clock: Any = time.monotonic) -> None:
        self.requests_per_minute = max(1, int(requests_per_minute))
        self.interval_seconds = 60.0 / self.requests_per_minute
        self._sleeper = sleeper
        self._clock = clock
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0
        self.acquire_count = 0
        self.total_wait_time = 0.0

    def acquire(self) -> float:
        with self._lock:
            now = self._clock()
            wait_seconds = max(0.0, self._next_allowed_at - now)
            if wait_seconds > 0:
                self._sleeper(wait_seconds)
                self.total_wait_time += wait_seconds
                now = self._clock()
            self._next_allowed_at = max(now, self._next_allowed_at) + self.interval_seconds
            self.acquire_count += 1
            return wait_seconds


class JQuantsApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        category: str = "api_error",
        endpoint: str = "",
        request_url: str = "",
        request_params: dict[str, str] | None = None,
        response_body: str = "",
        retry_after: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.category = category
        self.endpoint = endpoint
        self.request_url = request_url
        self.request_params = request_params or {}
        self.response_body = response_body
        self.retry_after = retry_after


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

    def __init__(
        self,
        env_path: Optional[Path] = None,
        timeout_seconds: int = 20,
        plan: str = "free",
        requests_per_minute: int = 5,
        parallel_fetch: bool = False,
        max_parallel_requests: int = 4,
    ) -> None:
        _load_env(env_path)
        api_key = os.getenv("JQUANTS_API_KEY")
        if not api_key:
            raise RuntimeError("JQUANTS_API_KEY が未設定です")

        self.api_key = api_key
        self.base_url = "https://api.jquants.com/v2"
        self.default_headers = {"x-api-key": self.api_key}
        self.timeout_seconds = timeout_seconds
        self.plan = normalize_jquants_plan(plan)
        self.capabilities = jquants_capabilities(self.plan)
        self.requests_per_minute = max(1, int(requests_per_minute))
        self.parallel_fetch = bool(parallel_fetch) and self.plan == "light"
        self.max_parallel_requests = max(1, int(max_parallel_requests))
        self.rate_limiter = RateLimiter(self.requests_per_minute)
        self.fetch_stats = {
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_fetch_time": 0.0,
            "rate_limit_wait_time": 0.0,
        }
        self.last_request_metadata: dict[str, Any] = {}

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

    def get_trading_calendar(self, start_date: date, end_date: date) -> list[dict[str, Any]]:
        return self._get_paginated_records(
            "/markets/calendar",
            {
                "from": start_date.strftime("%Y%m%d"),
                "to": end_date.strftime("%Y%m%d"),
            },
        )

    def service(self, cache_root: Path) -> "JQuantsDataService":
        return JQuantsDataService(self, cache_root)

    def get_topix_prices(self, start_date: date, end_date: Optional[date] = None) -> list[dict[str, Any]]:
        if not self.has_capability("topix_prices"):
            print("warning: J-Quants topix_prices is disabled for free plan; use Prime market average fallback.")
            return []
        params = {"from": start_date.isoformat()}
        if end_date:
            params["to"] = end_date.isoformat()
        return self._get_paginated_records("/indices/bars/daily/topix", params)

    def fetch_topix_prices(self, start_date: date, end_date: date) -> list[dict[str, Any]]:
        return normalize_topix_price_records(self.get_topix_prices(start_date, end_date))

    def fetch_topix_prices_cached(
        self,
        cache_root: Path,
        start_date: date,
        end_date: date,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        cache_path = cache_root / "jquants" / "topix_prices" / f"{start_date.isoformat()}_to_{end_date.isoformat()}.json"
        if cache_path.exists() and not force_refresh:
            records = _read_cached_json(cache_path).get("records", [])
            if _records_usable(records):
                _increment_fetch_stat(self, "cache_hits")
                return _cache_payload(records, cache_path, from_cache=True, available=True)
        if not self.has_capability("topix_prices"):
            return {
                "records": [],
                "cache_path": str(cache_path),
                "from_cache": False,
                "fallback_used": False,
                "warning": "topix_prices disabled for current J-Quants plan",
                "available": False,
                "saved": False,
                "usable": False,
            }
        try:
            _increment_fetch_stat(self, "cache_misses")
            records = self.fetch_topix_prices(start_date, end_date)
        except Exception as exc:
            if cache_path.exists():
                records = _read_cached_json(cache_path).get("records", [])
                if _records_usable(records):
                    _increment_fetch_stat(self, "cache_hits")
                    return _cache_payload(records, cache_path, from_cache=True, fallback_used=True, warning=str(exc), available=True)
            error_fields = _api_error_payload_fields(exc)
            return {
                "records": [],
                "cache_path": str(cache_path),
                "from_cache": False,
                "fallback_used": False,
                "warning": f"{exc}; empty_cache" if cache_path.exists() else str(exc),
                "available": False,
                "saved": False,
                "usable": False,
                "reason": "empty_cache" if cache_path.exists() else error_fields.get("reason", "api_error"),
                **error_fields,
            }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if _records_usable(records):
            _write_cache_records(cache_path, records)
        else:
            _record_empty_range(cache_root, "topix_prices", start_date, end_date, "empty_response")
        available = _records_usable(records)
        return {
            "records": records,
            "cache_path": str(cache_path),
            "from_cache": False,
            "fallback_used": False,
            "warning": "" if available else "api_success_but_empty",
            "available": available,
            "saved": available,
            "usable": available,
            "api_status": "200",
            "reason": "" if available else "empty_response",
        }

    def fetch_investor_types(self, start_date: date, end_date: date) -> list[dict[str, Any]]:
        if not self.has_capability("investor_types"):
            print("warning: J-Quants investor_types is disabled for free plan.")
            return []
        records = self._get_paginated_records(
            "/equities/investor-types",
            {
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
            },
        )
        return normalize_investor_type_records(records)

    def fetch_investor_types_cached(
        self,
        cache_root: Path,
        start_date: date,
        end_date: date,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        cache_path = cache_root / "jquants" / "investor_types" / f"{start_date.isoformat()}_to_{end_date.isoformat()}.json"
        if cache_path.exists() and not force_refresh:
            records = _read_cached_json(cache_path).get("records", [])
            if _records_usable(records):
                _increment_fetch_stat(self, "cache_hits")
                return _cache_payload(records, cache_path, from_cache=True, available=True)
        if not self.has_capability("investor_types"):
            return {
                "records": [],
                "cache_path": str(cache_path),
                "from_cache": False,
                "fallback_used": False,
                "warning": "investor_types disabled for current J-Quants plan",
                "available": False,
                "saved": False,
                "usable": False,
            }
        try:
            _increment_fetch_stat(self, "cache_misses")
            records = self.fetch_investor_types(start_date, end_date)
        except Exception as exc:
            if cache_path.exists():
                records = _read_cached_json(cache_path).get("records", [])
                if _records_usable(records):
                    _increment_fetch_stat(self, "cache_hits")
                    return _cache_payload(records, cache_path, from_cache=True, fallback_used=True, warning=str(exc), available=True)
            error_fields = _api_error_payload_fields(exc)
            return {
                "records": [],
                "cache_path": str(cache_path),
                "from_cache": False,
                "fallback_used": False,
                "warning": f"{exc}; empty_cache" if cache_path.exists() else str(exc),
                "available": False,
                "saved": False,
                "usable": False,
                "reason": "empty_cache" if cache_path.exists() else error_fields.get("reason", "api_error"),
                **error_fields,
            }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if _records_usable(records):
            _write_cache_records(cache_path, records)
        else:
            _record_empty_range(cache_root, "investor_types", start_date, end_date, "empty_response")
        available = _records_usable(records)
        return {
            "records": records,
            "cache_path": str(cache_path),
            "from_cache": False,
            "fallback_used": False,
            "warning": "" if available else "api_success_but_empty",
            "available": available,
            "saved": available,
            "usable": available,
            "api_status": "200",
            "reason": "" if available else "empty_response",
        }

    def fetch_earnings_calendar(self) -> list[dict[str, Any]]:
        records = self._get_paginated_records("/equities/earnings-calendar", {})
        return normalize_earnings_calendar_records(records)

    def fetch_earnings_calendar_cached(
        self,
        cache_root: Path,
        target_date: Optional[date] = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        cache_date = target_date or date.today()
        cache_path = cache_root / "jquants" / "earnings_calendar" / f"{cache_date.isoformat()}.json"
        if cache_path.exists() and not force_refresh:
            records = _read_cached_json(cache_path).get("records", [])
            if _records_usable(records):
                _increment_fetch_stat(self, "cache_hits")
                return _cache_payload(records, cache_path, from_cache=True, available=True, cache_date=cache_date)
        try:
            _increment_fetch_stat(self, "cache_misses")
            records = self.fetch_earnings_calendar()
        except Exception as exc:
            if cache_path.exists():
                records = _read_cached_json(cache_path).get("records", [])
                if _records_usable(records):
                    _increment_fetch_stat(self, "cache_hits")
                    return _cache_payload(records, cache_path, from_cache=True, fallback_used=True, warning=str(exc), available=True, cache_date=cache_date)
            error_fields = _api_error_payload_fields(exc)
            return {
                "records": [],
                "cache_path": str(cache_path),
                "cache_date": cache_date.isoformat(),
                "from_cache": False,
                "fallback_used": False,
                "warning": f"{exc}; empty_cache" if cache_path.exists() else str(exc),
                "filter_available": False,
                "available": False,
                "saved": False,
                "usable": False,
                "reason": "empty_cache" if cache_path.exists() else error_fields.get("reason", "api_error"),
                **error_fields,
            }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if _records_usable(records):
            _write_cache_records(cache_path, records)
        else:
            _record_empty_range(cache_root, "earnings_calendar", cache_date, cache_date, "empty_response")
        available = _records_usable(records)
        return {
            "records": records,
            "cache_path": str(cache_path),
            "cache_date": cache_date.isoformat(),
            "from_cache": False,
            "fallback_used": False,
            "warning": "" if available else "api_success_but_empty",
            "filter_available": available,
            "available": available,
            "saved": available,
            "usable": available,
            "api_status": "200",
            "reason": "" if available else "empty_response",
        }

    def fetch_earnings_calendar_period_cached(
        self,
        cache_root: Path,
        start_date: date,
        end_date: date,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        cache_path = cache_root / "jquants" / "earnings_calendar" / f"{start_date.isoformat()}_to_{end_date.isoformat()}.json"
        if cache_path.exists() and not force_refresh:
            records = _read_cached_json(cache_path).get("records", [])
            if _records_usable(records):
                _increment_fetch_stat(self, "cache_hits")
                return _cache_payload(records, cache_path, from_cache=True, available=True)
        try:
            _increment_fetch_stat(self, "cache_misses")
            records = [
                record
                for record in self.fetch_earnings_calendar()
                if _date_in_range(record.get("Date"), start_date, end_date)
            ]
        except Exception as exc:
            if cache_path.exists():
                records = _read_cached_json(cache_path).get("records", [])
                if _records_usable(records):
                    _increment_fetch_stat(self, "cache_hits")
                    return _cache_payload(records, cache_path, from_cache=True, fallback_used=True, warning=str(exc), available=True)
            error_fields = _api_error_payload_fields(exc)
            return {
                "records": [],
                "cache_path": str(cache_path),
                "from_cache": False,
                "fallback_used": False,
                "warning": f"{exc}; empty_cache" if cache_path.exists() else str(exc),
                "filter_available": False,
                "available": False,
                "saved": False,
                "usable": False,
                "reason": "empty_cache" if cache_path.exists() else error_fields.get("reason", "api_error"),
                **error_fields,
            }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if _records_usable(records):
            _write_cache_records(cache_path, records)
        else:
            _record_empty_range(cache_root, "earnings_calendar", start_date, end_date, "empty_response")
        available = _records_usable(records)
        return {
            "records": records,
            "cache_path": str(cache_path),
            "from_cache": False,
            "fallback_used": False,
            "warning": "" if available else "api_success_but_empty",
            "filter_available": available,
            "available": available,
            "saved": available,
            "usable": available,
            "api_status": "200",
            "reason": "" if available else "empty_response",
        }

    def fetch_financial_statements(self, start_date: date, end_date: date) -> list[dict[str, Any]]:
        if not self.has_capability("financial_statements"):
            print("warning: J-Quants financial_statements is disabled for current plan.")
            return []
        params: dict[str, str] = {}
        if start_date == end_date:
            params["date"] = start_date.strftime("%Y%m%d")
        return self._get_paginated_records("/fins/summary", params)

    def fetch_financial_statements_cached(
        self,
        cache_root: Path,
        start_date: date,
        end_date: date,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        cache_path = cache_root / "jquants" / "financial_statements" / f"{start_date.isoformat()}_to_{end_date.isoformat()}.json"
        if cache_path.exists() and not force_refresh:
            records = _read_cached_json(cache_path).get("records", [])
            if _records_usable(records):
                _increment_fetch_stat(self, "cache_hits")
                return _cache_payload(records, cache_path, from_cache=True, available=True)
        if not self.has_capability("financial_statements"):
            return {
                "records": [],
                "cache_path": str(cache_path),
                "from_cache": False,
                "fallback_used": False,
                "warning": "financial_statements disabled for current J-Quants plan",
                "available": False,
                "saved": False,
                "usable": False,
            }
        try:
            _increment_fetch_stat(self, "cache_misses")
            records = self.fetch_financial_statements(start_date, end_date)
        except Exception as exc:
            if cache_path.exists():
                records = _read_cached_json(cache_path).get("records", [])
                if _records_usable(records):
                    _increment_fetch_stat(self, "cache_hits")
                    return _cache_payload(records, cache_path, from_cache=True, fallback_used=True, warning=str(exc), available=True)
            error_fields = _api_error_payload_fields(exc)
            return {
                "records": [],
                "cache_path": str(cache_path),
                "from_cache": False,
                "fallback_used": False,
                "warning": f"{exc}; empty_cache" if cache_path.exists() else str(exc),
                "available": False,
                "saved": False,
                "usable": False,
                "reason": "empty_cache" if cache_path.exists() else error_fields.get("reason", "api_error"),
                **error_fields,
            }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if _records_usable(records):
            _write_cache_records(cache_path, records)
        else:
            _record_empty_range(cache_root, "financial_statements", start_date, end_date, "empty_response")
        available = _records_usable(records)
        return {
            "records": records,
            "cache_path": str(cache_path),
            "from_cache": False,
            "fallback_used": False,
            "warning": "" if available else "api_success_but_empty",
            "available": available,
            "saved": available,
            "usable": available,
            "api_status": "200",
            "reason": "" if available else "empty_response",
        }

    def get_stock_fundamentals(self, code: str) -> dict[str, Any]:
        # TODO: Fetch fundamentals/statements from J-Quants V2 API with x-api-key.
        raise NotImplementedError("J-Quants fundamentals fetch is not implemented yet.")

    def get_news(self, code: str) -> list[dict[str, Any]]:
        # J-Quants does not provide news. Use another API or web search in a later phase.
        return []

    def has_capability(self, capability: str) -> bool:
        return jquants_has_capability(self.plan, capability)

    def _build_request(self, path: str) -> Request:
        """Build a future J-Quants V2 HTTP request with x-api-key authentication."""
        return Request(f"{self.base_url}{path}", headers=self.default_headers)

    def _get_json(self, path: str) -> Any:
        request = self._build_request(path)
        self.last_request_metadata = {
            "endpoint": _endpoint_from_path(path),
            "url": getattr(request, "full_url", str(request)),
            "params": _params_from_path(path),
            "status_code": None,
            "response_body": "",
        }
        wait_seconds = self.rate_limiter.acquire()
        started_at = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                self.last_request_metadata["status_code"] = getattr(response, "status", 200)
                self.last_request_metadata["response_body"] = body[:500]
        except HTTPError as exc:
            body_summary = _http_error_body_summary(exc)
            category = _http_error_category(exc.code)
            if exc.code == 429:
                message = "J-Quants API rate limit exceeded. Wait a while and retry."
            else:
                message = f"J-Quants API request failed with HTTP {exc.code}."
            raise JQuantsApiError(
                message,
                status_code=exc.code,
                category=category,
                endpoint=_endpoint_from_path(path),
                request_url=getattr(request, "full_url", str(request)),
                request_params=_params_from_path(path),
                response_body=body_summary,
                retry_after=str(exc.headers.get("Retry-After") or ""),
            ) from exc
        except URLError as exc:
            raise JQuantsApiError(
                f"J-Quants network error: {exc.reason}",
                category="network_error",
                endpoint=_endpoint_from_path(path),
                request_url=getattr(request, "full_url", str(request)),
                request_params=_params_from_path(path),
            ) from exc
        except TimeoutError as exc:
            raise JQuantsApiError(
                "J-Quants network error: request timed out.",
                category="timeout",
                endpoint=_endpoint_from_path(path),
                request_url=getattr(request, "full_url", str(request)),
                request_params=_params_from_path(path),
            ) from exc
        finally:
            elapsed = time.perf_counter() - started_at
            _increment_fetch_stat(self, "api_calls")
            _increment_fetch_stat(self, "total_fetch_time", elapsed)
            _increment_fetch_stat(self, "rate_limit_wait_time", wait_seconds)

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("J-Quants response format is invalid JSON.") from exc

    def _get_paginated_records(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        pagination_key = ""
        first_request_metadata: dict[str, Any] | None = None
        while True:
            request_params = dict(params)
            if pagination_key:
                request_params["pagination_key"] = pagination_key
            payload = self._get_json(f"{path}?{urlencode(request_params)}")
            if first_request_metadata is None:
                first_request_metadata = dict(self.last_request_metadata)
            records.extend(_extract_records(payload))
            if not isinstance(payload, dict):
                break
            pagination_key = str(payload.get("pagination_key") or "")
            if not pagination_key:
                break
        if first_request_metadata is not None:
            self.last_request_metadata = first_request_metadata
            self.last_request_metadata["records"] = len(records)
        return records


class JQuantsDataService:
    """Reusable J-Quants fetch/cache boundary for application and ML callers."""

    def __init__(self, provider: JQuantsDataProvider, cache_root: Path) -> None:
        self.provider = provider
        self.cache_root = cache_root

    def cache_path(self, endpoint: str, filename: str) -> Path:
        spec = JQUANTS_ENDPOINTS[endpoint]
        return self.cache_root / "jquants" / spec["cache_dir"] / filename

    def fetch_listed_info_cached(self, target_date: date, force_refresh: bool = False) -> dict[str, Any]:
        return self._fetch_cached(
            endpoint="listed_info",
            cache_path=self.cache_path("listed_info", f"{target_date.isoformat()}.json"),
            start_date=target_date,
            end_date=target_date,
            fetcher=self.provider.get_listed_stocks,
            force_refresh=force_refresh,
        )

    def fetch_daily_prices_cached(self, target_date: date, force_refresh: bool = False) -> dict[str, Any]:
        return self._fetch_cached(
            endpoint="prices",
            cache_path=self.cache_path("prices", f"{target_date.isoformat()}.json"),
            start_date=target_date,
            end_date=target_date,
            fetcher=lambda: self.provider.get_daily_prices(target_date),
            force_refresh=force_refresh,
        )

    def fetch_trading_calendar_cached(self, start_date: date, end_date: date, force_refresh: bool = False) -> dict[str, Any]:
        return self._fetch_cached(
            endpoint="trading_calendar",
            cache_path=self.cache_path("trading_calendar", f"{start_date.isoformat()}_to_{end_date.isoformat()}.json"),
            start_date=start_date,
            end_date=end_date,
            fetcher=lambda: self.provider.get_trading_calendar(start_date, end_date),
            force_refresh=force_refresh,
        )

    def fetch_topix_prices_cached(self, start_date: date, end_date: date, force_refresh: bool = False) -> dict[str, Any]:
        return self.provider.fetch_topix_prices_cached(self.cache_root, start_date, end_date, force_refresh)

    def fetch_investor_types_cached(self, start_date: date, end_date: date, force_refresh: bool = False) -> dict[str, Any]:
        return self.provider.fetch_investor_types_cached(self.cache_root, start_date, end_date, force_refresh)

    def fetch_earnings_calendar_cached(self, target_date: date, force_refresh: bool = False) -> dict[str, Any]:
        return self.provider.fetch_earnings_calendar_cached(self.cache_root, target_date, force_refresh)

    def fetch_earnings_calendar_period_cached(self, start_date: date, end_date: date, force_refresh: bool = False) -> dict[str, Any]:
        return self.provider.fetch_earnings_calendar_period_cached(self.cache_root, start_date, end_date, force_refresh)

    def fetch_financial_statements_cached(self, start_date: date, end_date: date, force_refresh: bool = False) -> dict[str, Any]:
        return self.provider.fetch_financial_statements_cached(self.cache_root, start_date, end_date, force_refresh)

    def _fetch_cached(
        self,
        endpoint: str,
        cache_path: Path,
        start_date: date,
        end_date: date,
        fetcher: Any,
        force_refresh: bool,
    ) -> dict[str, Any]:
        if cache_path.exists() and not force_refresh:
            records = _read_cached_json(cache_path).get("records", [])
            if _records_usable(records):
                _increment_fetch_stat(self.provider, "cache_hits")
                return _cache_payload(records, cache_path, from_cache=True, available=True)
        try:
            _increment_fetch_stat(self.provider, "cache_misses")
            records = fetcher()
        except Exception as exc:
            if cache_path.exists():
                records = _read_cached_json(cache_path).get("records", [])
                if _records_usable(records):
                    _increment_fetch_stat(self.provider, "cache_hits")
                    return _cache_payload(records, cache_path, from_cache=True, fallback_used=True, warning=str(exc), available=True)
            error_fields = _api_error_payload_fields(exc)
            return {
                "records": [],
                "cache_path": str(cache_path),
                "from_cache": False,
                "fallback_used": False,
                "warning": f"{exc}; empty_cache" if cache_path.exists() else str(exc),
                "available": False,
                "saved": False,
                "usable": False,
                "reason": "empty_cache" if cache_path.exists() else error_fields.get("reason", "api_error"),
                **error_fields,
            }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if _records_usable(records):
            _write_cache_records(cache_path, records)
        else:
            _record_empty_range(self.cache_root, endpoint, start_date, end_date, "empty_response")
        available = _records_usable(records)
        return {
            "records": records,
            "cache_path": str(cache_path),
            "from_cache": False,
            "fallback_used": False,
            "warning": "" if available else "api_success_but_empty",
            "available": available,
            "saved": available,
            "usable": available,
            "api_status": "200",
            "reason": "" if available else "empty_response",
            "request_url": (getattr(self.provider, "last_request_metadata", {}) or {}).get("url", ""),
            "request_params": (getattr(self.provider, "last_request_metadata", {}) or {}).get("params", {}),
            "http_status": (getattr(self.provider, "last_request_metadata", {}) or {}).get("status_code", ""),
            "response_body": (getattr(self.provider, "last_request_metadata", {}) or {}).get("response_body", ""),
        }


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


def normalize_topix_price_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        row = {
            "date": _format_date(str(_first_value(record, ["Date", "date"]))),
            "open": _number_value(record, ["Open", "open", "O"]),
            "high": _number_value(record, ["High", "high", "H"]),
            "low": _number_value(record, ["Low", "low", "L"]),
            "close": _number_value(record, ["Close", "close", "C"]),
        }
        if row["date"] and row["close"] is not None:
            normalized.append(row)
    return normalized


def _first_value(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if record.get(key) is not None:
            return record.get(key)
    return ""


def _number_value(record: dict[str, Any], keys: list[str]) -> float | None:
    value = _first_value(record, keys)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_date(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def _date_in_range(value: Any, start_date: date, end_date: date) -> bool:
    try:
        parsed = date.fromisoformat(_format_date(str(value or "")))
    except ValueError:
        return False
    return start_date <= parsed <= end_date


def _read_cached_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_cached_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_cache_records(path: Path, records: list[dict[str, Any]]) -> None:
    _write_cached_json(
        path,
        {
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "records": records,
        },
    )


def _record_empty_range(cache_root: Path, endpoint: str, start_date: date, end_date: date, reason: str) -> None:
    path = cache_root / "jquants" / "empty_ranges.json"
    try:
        payload = _read_cached_json(path) if path.exists() else {}
    except Exception:
        payload = {}
    ranges = list(payload.get(endpoint, [])) if isinstance(payload, dict) else []
    entry = {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "reason": reason,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    if not any(item.get("start") == entry["start"] and item.get("end") == entry["end"] for item in ranges if isinstance(item, dict)):
        ranges.append(entry)
    payload[endpoint] = ranges
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_cached_json(path, payload)


def _records_usable(records: Any) -> bool:
    return isinstance(records, list) and len(records) > 0


def _cache_payload(
    records: list[dict[str, Any]],
    cache_path: Path,
    from_cache: bool,
    fallback_used: bool = False,
    warning: str = "",
    available: bool = True,
    cache_date: date | None = None,
) -> dict[str, Any]:
    payload = {
        "records": records,
        "cache_path": str(cache_path),
        "from_cache": from_cache,
        "fallback_used": fallback_used,
        "warning": warning,
        "available": available,
        "saved": False,
        "usable": _records_usable(records),
        "reason": "" if _records_usable(records) else "empty_cache",
    }
    if cache_date is not None:
        payload["cache_date"] = cache_date.isoformat()
        payload["filter_available"] = available
    return payload


def _increment_fetch_stat(provider: Any, key: str, amount: float = 1.0) -> None:
    stats = getattr(provider, "fetch_stats", None)
    if isinstance(stats, dict):
        stats[key] = stats.get(key, 0) + amount


def _http_error_category(status_code: int) -> str:
    if status_code in {401, 403}:
        return "auth_or_plan_error"
    if status_code == 400:
        return "bad_request"
    if status_code == 404:
        return "endpoint_not_found"
    if status_code == 429:
        return "rate_limit"
    return "api_error"


def _http_error_body_summary(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    return body.replace("\n", " ")[:500]


def _api_error_payload_fields(exc: Exception) -> dict[str, Any]:
    if not isinstance(exc, JQuantsApiError):
        return {"api_status": "api_error", "reason": "api_error"}
    return {
        "api_status": exc.category,
        "reason": exc.category,
        "http_status": exc.status_code,
        "request_url": exc.request_url,
        "request_params": exc.request_params,
        "response_body": exc.response_body,
        "retry_after": exc.retry_after,
    }


def _endpoint_from_path(path: str) -> str:
    return urlsplit(path).path


def _params_from_path(path: str) -> dict[str, str]:
    return dict(parse_qsl(urlsplit(path).query))
