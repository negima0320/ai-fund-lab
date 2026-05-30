"""News providers used for lightweight stock news scoring."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree


class BaseNewsProvider(ABC):
    @abstractmethod
    def get_news(self, code: str, name: str, target_date: str) -> dict[str, Any]:
        """Return cached or fetched news payload for one stock."""


class DummyNewsProvider(BaseNewsProvider):
    def get_news(self, code: str, name: str, target_date: str) -> dict[str, Any]:
        return {
            "code": code,
            "name": name,
            "date": target_date,
            "provider": "dummy",
            "query": "",
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "articles": [],
            "limitation": "dummy providerのためニュース取得なし",
        }


class GoogleNewsRSSProvider(BaseNewsProvider):
    def __init__(self, root: Path, config: dict[str, Any]) -> None:
        self.root = root
        news_config = config.get("news", {})
        self.lookback_days = int(news_config.get("lookback_days", 7))
        self.cache_enabled = bool(news_config.get("cache_enabled", True))
        self.cache_dir = root / "data" / "raw" / "news"

    def get_news(self, code: str, name: str, target_date: str) -> dict[str, Any]:
        cache_path = self.cache_dir / f"news_{target_date}_{code}.json"
        if self.cache_enabled and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        query = f"{name} 株 OR {name} 決算 OR {code} 株"
        limitation = "Google News RSSは過去日付指定の再現性が限定的なため、現時点では最新ニュースを取得してdate基準で可能な範囲だけフィルタします。"
        try:
            articles = self._fetch_articles(query, target_date)
        except Exception as exc:  # RSS失敗でスコアリング全体を止めない
            payload = {
                "code": code,
                "name": name,
                "date": target_date,
                "provider": "google_news_rss",
                "query": query,
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "articles": [],
                "limitation": f"ニュース取得に失敗: {exc}",
            }
            self._write_cache(cache_path, payload)
            return payload

        payload = {
            "code": code,
            "name": name,
            "date": target_date,
            "provider": "google_news_rss",
            "query": query,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "articles": articles,
            "limitation": limitation,
        }
        self._write_cache(cache_path, payload)
        return payload

    def _fetch_articles(self, query: str, target_date: str) -> list[dict[str, Any]]:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=ja&gl=JP&ceid=JP:ja"
        request = Request(url, headers={"User-Agent": "ai-fund-lab/0.1"})
        with urlopen(request, timeout=15) as response:
            body = response.read()
        root = ElementTree.fromstring(body)
        target = date.fromisoformat(target_date)
        start = target - timedelta(days=self.lookback_days)
        articles = []
        for item in root.findall("./channel/item"):
            published_at = _text(item, "pubDate")
            published_date = _parse_rss_date(published_at)
            if published_date and not (start <= published_date <= target):
                continue
            source = item.find("source")
            articles.append(
                {
                    "title": _text(item, "title"),
                    "link": _text(item, "link"),
                    "published_at": published_at,
                    "source": source.text if source is not None and source.text else "",
                }
            )
        return articles

    def _write_cache(self, path: Path, payload: dict[str, Any]) -> None:
        if not self.cache_enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_news_provider(config: dict[str, Any], root: Path) -> BaseNewsProvider:
    news_config = config.get("news", {})
    if not news_config.get("enabled", False):
        return DummyNewsProvider()
    provider = news_config.get("provider", "dummy")
    if provider == "google_news_rss":
        return GoogleNewsRSSProvider(root, config)
    return DummyNewsProvider()


def _text(element: ElementTree.Element, tag: str) -> str:
    child = element.find(tag)
    return child.text.strip() if child is not None and child.text else ""


def _parse_rss_date(value: str) -> date | None:
    if not value:
        return None
    try:
        parsed = datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %Z")
        return parsed.replace(tzinfo=timezone.utc).date()
    except ValueError:
        return None
