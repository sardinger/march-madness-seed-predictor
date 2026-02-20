#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup, Comment
from dotenv import load_dotenv
from pymongo import MongoClient


URL = "https://www.sports-reference.com/cbb/seasons/men/2026-ratings.html"
DB_NAME = "march-madness"
COLLECTION_NAME = "season-ratings-2026"

MAX_ROWS: Optional[int] = 100  # set to 25 for top 25


TABLE_ID_CANDIDATES = ["ratings", "schools", "basic_school", "ratings_school"]


def convert_value(value: Optional[str]) -> Any:
    if value is None:
        return None
    v = value.strip()
    if v in ("", "-", "NA", "N/A"):
        return None
    v = v.replace(",", "")
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def convert_team_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "-")
        .replace("(", "")
        .replace(")", "")
        .replace("'", "")
        .replace("&", "")
        .replace(".", "")
    )


def fetch_html(url: str, *, timeout: int = 20) -> str:
    """
    Polite fetch. If Sports-Reference blocks (403), we stop rather than bypass.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; cbb-scraper/1.0; +https://example.com)",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }
    r = requests.get(url, headers=headers, timeout=timeout)

    if r.status_code in (403, 429):
        raise RuntimeError(
            f"Blocked by Sports-Reference (HTTP {r.status_code}). "
            "Don’t try to bypass. Instead, save the page HTML in your browser and parse locally."
        )

    r.raise_for_status()
    return r.text


def find_table_sportsref(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """
    Sports-Reference sometimes puts tables inside HTML comments.
    This tries:
      1) normal soup.find('table', id=...)
      2) search inside comments for a table with an expected id
      3) fallback: first table found in any comment
    """
    # 1) direct
    for tid in TABLE_ID_CANDIDATES:
        t = soup.find("table", id=tid)
        if t:
            return t

    # 2) inside comments
    comments = soup.find_all(string=lambda s: isinstance(s, Comment))
    for c in comments:
        c_str = str(c)
        for tid in TABLE_ID_CANDIDATES:
            if f'id="{tid}"' in c_str:
                csoup = BeautifulSoup(c_str, "html.parser")
                t = csoup.find("table", id=tid)
                if t:
                    return t

    # 3) fallback: first commented table
    for c in comments:
        csoup = BeautifulSoup(str(c), "html.parser")
        t = csoup.find("table")
        if t:
            return t

    return None


def parse_ratings(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    table = find_table_sportsref(soup)
    if not table:
        raise RuntimeError("Could not find ratings table on the page (or in comments).")

    docs: List[Dict[str, Any]] = []
    tbody = table.find("tbody")
    if not tbody:
        raise RuntimeError("Table found, but no <tbody>.")

    for tr in tbody.find_all("tr"):
        # skip header/separator rows
        if tr.get("class") and "thead" in tr.get("class", []):
            continue

        doc: Dict[str, Any] = {"season": 2026}

        # Sports-Reference uses 'th' for rank/school sometimes
        for cell in tr.find_all(["th", "td"]):
            key = cell.get("data-stat")
            if not key:
                continue

            text = cell.get_text(strip=True)

            # normalize a couple common ones
            if key in ("rk", ""):
                key = "rank"
            if key == "school_name":
                key = "school"

            doc[key] = convert_value(text)

        # enforce max rows
        if doc.get("school") or doc.get("school_name"):
            docs.append(doc)

        if MAX_ROWS is not None and len(docs) >= MAX_ROWS:
            break

    return docs


def main() -> None:
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("Missing MONGODB_URI in environment (.env)")

    client = MongoClient(uri)
    col = client[DB_NAME][COLLECTION_NAME]

    html = fetch_html(URL)
    docs = parse_ratings(html)

    upserted = 0
    replaced = 0

    top_100_li = []
    for d in docs:
        school = d.get("school")
        if not school:
            continue

        top_100_li.append(convert_team_name(school))

        res = col.replace_one({"season": d["season"], "school": school}, d, upsert=True)
        if res.upserted_id is not None:
            upserted += 1
        elif res.matched_count:
            replaced += 1

    print(top_100_li)  # Copy/paste this list into the constants file
    print(f"Done. parsed={len(docs)} upserted={upserted} replaced={replaced}")


if __name__ == "__main__":
    main()
