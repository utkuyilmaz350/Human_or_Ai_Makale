from __future__ import annotations

import argparse
import time
from typing import Any

import feedparser
import pandas as pd
import requests


def fetch_arxiv(query: str, start: int, max_results: int) -> dict[str, Any]:
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    return {"entries": feed.entries}


def collect(query: str, max_results: int, batch_size: int = 100, sleep_s: float = 1.0) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start = 0
    while len(rows) < max_results:
        take = min(batch_size, max_results - len(rows))
        data = fetch_arxiv(query=query, start=start, max_results=take)
        entries = data.get("entries", [])
        if not entries:
            break

        for e in entries:
            rows.append(
                {
                    "id": e.get("id"),
                    "title": (e.get("title") or "").strip().replace("\n", " "),
                    "summary": (e.get("summary") or "").strip().replace("\n", " "),
                    "published": e.get("published"),
                    "updated": e.get("updated"),
                }
            )

        start += len(entries)
        time.sleep(sleep_s)

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["summary"]).drop_duplicates(subset=["summary"]).reset_index(drop=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help='arXiv query. Örn: "cat:cs.CL" veya "all:transformer"')
    p.add_argument("--max-results", type=int, default=3000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = collect(query=args.query, max_results=args.max_results)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
