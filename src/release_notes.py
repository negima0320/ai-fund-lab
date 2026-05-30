"""Generate release notes from git commit history."""

from __future__ import annotations

import subprocess
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path
from typing import Any


CATEGORY_ORDER = [
    "機能追加",
    "修正",
    "ドキュメント",
    "テスト",
    "リファクタリング",
    "その他",
]

CATEGORY_PREFIXES = {
    "機能追加": ("feat", "add"),
    "修正": ("fix", "bugfix", "hotfix"),
    "ドキュメント": ("docs", "doc"),
    "テスト": ("test", "tests"),
    "リファクタリング": ("refactor", "refactoring"),
    "その他": ("chore", "ci", "build", "style", "perf", "release"),
}


def generate_release_notes(since: str, until: str, repo_root: Path) -> dict[str, Any]:
    since_date = _parse_date(since, "since")
    until_date = _parse_date(until, "until")
    if since_date > until_date:
        raise ValueError("--since must be earlier than or equal to --until.")

    commits = get_commits_between(since_date, until_date, repo_root)
    categories = {category: [] for category in CATEGORY_ORDER}
    for commit in commits:
        category = classify_commit(commit["message"])
        commit["category"] = category
        categories[category].append(commit)

    return {
        "since": since,
        "until": until,
        "total_commits": len(commits),
        "categories": {
            category: {
                "count": len(items),
                "commits": items,
            }
            for category, items in categories.items()
        },
        "commits": commits,
        "comment": generate_rookie_release_comment(commits),
    }


def get_commits_between(since: date, until: date, repo_root: Path) -> list[dict[str, str]]:
    since_at = datetime.combine(since, time.min).isoformat(timespec="seconds")
    until_at = datetime.combine(until, time.max).isoformat(timespec="seconds")
    command = [
        "git",
        "log",
        f"--since={since_at}",
        f"--until={until_at}",
        "--date=iso-strict",
        "--pretty=format:%h%x09%ad%x09%s",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git コマンドが見つかりません。") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"git log の取得に失敗しました: {message}") from exc

    commits = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        commit_hash, committed_at, message = parts
        commits.append(
            {
                "hash": commit_hash,
                "datetime": committed_at,
                "message": message,
            }
        )
    return commits


def classify_commit(message: str) -> str:
    normalized = message.strip().lower()
    prefix = normalized.split(":", 1)[0].split("(", 1)[0].strip()
    for category in CATEGORY_ORDER:
        if prefix in CATEGORY_PREFIXES[category]:
            return category
    return "その他"


def render_release_notes_markdown(notes: dict[str, Any]) -> str:
    lines = [
        "# AI Fund Lab 開発ノート",
        "",
        "## 対象期間",
        f"{notes['since']} 〜 {notes['until']}",
        "",
        "## 変更サマリ",
        "",
    ]
    categories = notes["categories"]
    for category in CATEGORY_ORDER:
        lines.append(f"### {category}")
        commits = categories.get(category, {}).get("commits", [])
        if not commits:
            lines.append("- 該当なし")
        else:
            for commit in commits:
                lines.append(f"- `{commit['hash']}` {commit['message']}")
        lines.append("")

    category_counts = {category: categories.get(category, {}).get("count", 0) for category in CATEGORY_ORDER}
    lines.extend(
        [
            "## 主な進捗",
            "",
            f"- コミット件数: {notes['total_commits']}",
        ]
    )
    for category in CATEGORY_ORDER:
        lines.append(f"- {category}: {category_counts[category]}件")

    lines.extend(
        [
            "",
            "## 新人ディーラー1号コメント",
            "",
            notes["comment"],
            "",
        ]
    )
    return "\n".join(lines)


def generate_rookie_release_comment(commits: list[dict[str, str]]) -> str:
    if not commits:
        return "本期間の開発コミットはありません。感情は考慮せず、次回の記録を待ちます。"

    counts = Counter(commit.get("category", "その他") for commit in commits)
    if counts["機能追加"] >= max(counts.values()):
        return "本期間は機能追加が中心でした。運用記録と検証機能が拡張され、実売買に向けた準備が段階的に進んでいます。"
    if counts["修正"] >= max(counts.values()):
        return "本期間は修正作業が中心でした。既存機能の安定性を優先し、規律ある改善が進んでいます。"
    if counts["テスト"] > 0:
        return "本期間はテスト整備が進みました。感情は考慮せず、再現可能な検証体制を優先します。"
    if counts["ドキュメント"] > 0:
        return "本期間はドキュメント整備が進みました。判断根拠を残す方針に沿った、堅実な開発です。"
    return "本期間は基盤整備が進みました。実売買に向けた準備として、規律ある開発が継続されています。"


def _parse_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"--{label} must be in YYYY-MM-DD format.") from exc
