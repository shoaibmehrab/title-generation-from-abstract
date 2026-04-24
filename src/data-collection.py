import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def invert_abstract(inv_idx: Optional[Dict[str, List[int]]]) -> Optional[str]:
    if not inv_idx:
        return None

    positions = [p for pos_list in inv_idx.values() for p in pos_list]
    if not positions:
        return None

    max_pos = max(positions)
    words = [""] * (max_pos + 1)

    for word, pos_list in inv_idx.items():
        for pos in pos_list:
            words[pos] = word

    abstract = " ".join(w for w in words if w).strip()
    return abstract or None


def sanitize_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]+", "", text)
    return text[:80] or "query"


def short_openalex_id(openalex_url: Optional[str]) -> Optional[str]:
    if not openalex_url:
        return None
    return openalex_url.rstrip("/").split("/")[-1]


def build_filter(
    from_date: Optional[str],
    to_date: Optional[str],
    topic_ids: List[str],
    concept_ids: List[str],
) -> str:
    parts = []

    if from_date:
        parts.append(f"from_publication_date:{from_date}")
    if to_date:
        parts.append(f"to_publication_date:{to_date}")

    # OR logic inside each filter: id1|id2|id3
    if topic_ids:
        parts.append("topics.id:" + "|".join(topic_ids))
    if concept_ids:
        parts.append("concepts.id:" + "|".join(concept_ids))

    # AND across filter blocks with comma
    return ",".join(parts)


def fetch_works(
    query: str,
    filter_expr: str,
    mailto: str,
    per_page: int,
    max_results: int,
) -> List[Dict]:
    select_fields = (
        "id,title,display_name,abstract_inverted_index,publication_year,topics,concepts"
    )

    cursor = "*"
    all_works: List[Dict] = []

    while len(all_works) < max_results:
        params = {
            "search": query,
            "select": select_fields,
            "per-page": per_page,
            "cursor": cursor,
            "mailto": mailto,
        }
        if filter_expr:
            params["filter"] = filter_expr

        resp = requests.get(OPENALEX_WORKS_URL, params=params, timeout=45)
        resp.raise_for_status()
        payload = resp.json()

        batch = payload.get("results", [])
        if not batch:
            break

        remaining = max_results - len(all_works)
        all_works.extend(batch[:remaining])

        next_cursor = payload.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = next_cursor

    # Deduplicate by OpenAlex URL id
    unique_works = []
    seen = set()
    for w in all_works:
        wid = w.get("id")
        if not wid or wid in seen:
            continue
        seen.add(wid)
        unique_works.append(w)

    return unique_works


def normalize_work(work: Dict, keep_empty_abstract: bool = False) -> Optional[Dict]:
    title = (work.get("title") or work.get("display_name") or "").strip()
    abstract = (invert_abstract(work.get("abstract_inverted_index")) or "").strip()

    # By default, skip empty abstracts
    if (not abstract) and (not keep_empty_abstract):
        return None

    topics = work.get("topics") or []
    topic_names = [t.get("display_name") for t in topics if t.get("display_name")]
    topic_ids_short = [
        short_openalex_id(t.get("id")) for t in topics if t.get("id")
    ]

    # Optional: keep concept names for analysis/debugging
    concepts = work.get("concepts") or []
    concept_names = [c.get("display_name") for c in concepts if c.get("display_name")]

    primary_topic = None
    if topics:
        primary_topic = topics[0].get("display_name")

    return {
        "id": short_openalex_id(work.get("id")),  # changed from openalex_id -> id
        "title": title,
        "abstract": abstract,
        "publication_year": work.get("publication_year"),
        "topics": topic_names,          # actual topic names
        "primary_topic": primary_topic,
    }


def save_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_csv(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Removed columns you said you do not need:
    # text, doi, publication_date, query, source, collected_at_utc
    fields = [
        "id",
        "title",
        "abstract",
        "publication_year",
        "topics",
        "primary_topic",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            quoting=csv.QUOTE_ALL,  # robust CSV escaping for commas/newlines
        )
        writer.writeheader()
        for rec in records:
            row = rec.copy()
            row["topics"] = json.dumps(row["topics"], ensure_ascii=False)
            writer.writerow(row)


def save_metadata(meta: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect papers from OpenAlex and save raw/processed data."
    )
    parser.add_argument("--query", default="natural language processing")
    parser.add_argument("--from-date", default="2020-01-01")
    parser.add_argument("--to-date", default="2026-03-31")
    parser.add_argument("--topic-id", action="append", default=[])
    parser.add_argument("--concept-id", action="append", default=[])
    parser.add_argument("--per-page", type=int, default=50)
    parser.add_argument("--max-results", type=int, default=12000)
    parser.add_argument("--mailto", default="your_email@example.com")
    parser.add_argument(
        "--output-root",
        default="/home/shoaib/nlp-sp26/project/Data",
        help="Root folder for output structure",
    )
    parser.add_argument(
        "--keep-empty-abstract",
        action="store_true",
        help="Keep rows where abstract is empty (default: skip them)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    filter_expr = build_filter(
        from_date=args.from_date,
        to_date=args.to_date,
        topic_ids=args.topic_id,
        concept_ids=args.concept_id,
    )

    works = fetch_works(
        query=args.query,
        filter_expr=filter_expr,
        mailto=args.mailto,
        per_page=args.per_page,
        max_results=args.max_results,
    )

    normalized: List[Dict] = []
    for w in works:
        rec = normalize_work(w, keep_empty_abstract=args.keep_empty_abstract)
        if rec is not None:
            normalized.append(rec)

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    slug = sanitize_slug(args.query)
    run_name = f"{stamp}_{slug}"

    output_root = Path(args.output_root)

    raw_path = output_root / "raw" / "openalex" / f"{run_name}.jsonl"
    processed_path = output_root / "processed" / "openalex" / f"{run_name}.csv"
    metadata_path = output_root / "metadata" / "openalex" / f"{run_name}.json"

    save_jsonl(normalized, raw_path)
    save_csv(normalized, processed_path)

    metadata = {
        "run_name": run_name,
        "created_at_utc": now.isoformat(),
        "query": args.query,
        "filter": filter_expr,
        "per_page": args.per_page,
        "max_results_requested": args.max_results,
        "records_saved": len(normalized),
        "empty_abstract_policy": "kept" if args.keep_empty_abstract else "dropped",
        "paths": {
            "raw_jsonl": str(raw_path),
            "processed_csv": str(processed_path),
            "metadata_json": str(metadata_path),
        },
    }
    save_metadata(metadata, metadata_path)

    print(f"Collected {len(normalized)} papers")
    print(f"Raw JSONL: {raw_path}")
    print(f"Processed CSV: {processed_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()