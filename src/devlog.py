"""Development log section for note drafts."""

from __future__ import annotations

import subprocess
from datetime import date, datetime, time
from pathlib import Path
from typing import Optional


def generate_devlog_section(article_date: str, repo_root: Path) -> str:
    commits = get_daily_commits(article_date, repo_root)
    if commits is None:
        return "## 今日の開発内容\n\n開発ログを取得できませんでした。"
    if not commits:
        return "## 今日の開発内容\n\n本日の開発コミットはありません。"

    lines = ["## 今日の開発内容", ""]
    for commit in commits:
        lines.append(f"- {commit['time']} `{commit['hash']}` {commit['message']}")
    return "\n".join(lines)


def get_daily_commits(article_date: str, repo_root: Path) -> Optional[list[dict[str, str]]]:
    try:
        day = date.fromisoformat(article_date)
    except ValueError:
        return None

    since = datetime.combine(day, time.min).isoformat(timespec="seconds")
    until = datetime.combine(day, time.max).isoformat(timespec="seconds")
    command = [
        "git",
        "log",
        f"--since={since}",
        f"--until={until}",
        "--date=format:%H:%M",
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
    except (OSError, subprocess.CalledProcessError):
        return None

    commits = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        commit_hash, commit_time, message = parts
        commits.append({"hash": commit_hash, "time": commit_time, "message": message})
    return commits
