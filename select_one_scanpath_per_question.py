#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select one participant scanpath per (image_id, question_id).")
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--qa-jsonl", type=Path, default=None,
                   help="Optional canonical QA JSONL used to prefer answer-matching participant rows.")
    p.add_argument("--strategy", choices=["median_length", "random", "shortest", "longest", "first"],
                   default="median_length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-scanpath-length", type=int, default=1)
    p.add_argument("--max-scanpath-length", type=int, default=None)
    p.add_argument("--prefer-not-truncated", action="store_true")
    p.add_argument("--strict-canonical-answer", action="store_true")
    p.add_argument("--stats-json", type=Path, default=None)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_line_number"] = line_no
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().casefold().replace("%", " percent ")
    text = re.sub(r"[\s,_]+", " ", text)
    text = re.sub(r"[^\w.\-+ ]+", "", text)
    return text.strip()


def extract_answer(row: dict[str, Any]) -> str:
    if row.get("answer") is not None:
        return str(row["answer"])
    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", ""))
    answers = row.get("answers")
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return str(answers or "")


def group_key(row: dict[str, Any]) -> tuple[str, str]:
    image_id = str(row.get("image_id") or "").strip()
    question_id = str(row.get("question_id") or "").strip()
    if image_id and question_id:
        return image_id, question_id
    sample_id = str(row.get("sample_id") or "")
    m = re.match(r"(.+?)_(Q\d+)(?:_|$)", sample_id)
    if m:
        return m.group(1), m.group(2)
    raise KeyError(f"Cannot determine group key for sample_id={sample_id!r}")


def build_canonical_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    out = {}
    for row in rows:
        try:
            key = group_key(row)
        except KeyError:
            continue
        ans = extract_answer(row)
        if ans:
            out[key] = ans
    return out


def answer_matches(candidate: str, canonical: str) -> bool:
    a, b = normalize_text(candidate), normalize_text(canonical)
    if a == b:
        return True

    def to_number(text: str) -> float | None:
        compact = text.replace(" percent", "").replace(" ", "")
        try:
            return float(compact)
        except ValueError:
            return None

    na, nb = to_number(a), to_number(b)
    return na is not None and nb is not None and abs(na - nb) <= 1e-6


def choose_row(rows: list[dict[str, Any]], strategy: str, rng: random.Random) -> dict[str, Any]:
    if strategy == "first":
        return min(rows, key=lambda r: int(r["_source_line_number"]))
    if strategy == "random":
        return rng.choice(rows)
    ordered = sorted(rows, key=lambda r: (
        int(r.get("scanpath_length", 0)),
        str(r.get("participant_id", "")),
        int(r["_source_line_number"]),
    ))
    if strategy == "shortest":
        return ordered[0]
    if strategy == "longest":
        return ordered[-1]
    return ordered[(len(ordered) - 1) // 2]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_rows = load_jsonl(args.input_jsonl)
    canonical = build_canonical_lookup(load_jsonl(args.qa_jsonl)) if args.qa_jsonl else {}

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    removed_by_length = 0
    for row in input_rows:
        length = int(row.get("scanpath_length", 0))
        if length < args.min_scanpath_length:
            removed_by_length += 1
            continue
        if args.max_scanpath_length is not None and length > args.max_scanpath_length:
            removed_by_length += 1
            continue
        grouped[group_key(row)].append(row)

    selected = []
    stats = {
        "input_rows": len(input_rows),
        "unique_question_groups": len(grouped),
        "removed_by_length": removed_by_length,
        "groups_with_canonical_answer": 0,
        "groups_filtered_to_answer_matches": 0,
        "groups_without_matching_answer": 0,
        "groups_dropped": 0,
        "groups_preferred_not_truncated": 0,
    }

    for key in sorted(grouped):
        candidates = grouped[key]
        canonical_answer = canonical.get(key)

        if canonical_answer is not None:
            stats["groups_with_canonical_answer"] += 1
            matched = [r for r in candidates if answer_matches(extract_answer(r), canonical_answer)]
            if matched:
                candidates = matched
                stats["groups_filtered_to_answer_matches"] += 1
            else:
                stats["groups_without_matching_answer"] += 1
                if args.strict_canonical_answer:
                    stats["groups_dropped"] += 1
                    continue

        if args.prefer_not_truncated and args.max_scanpath_length is not None:
            untruncated = [r for r in candidates if int(r.get("scanpath_length", 0)) < args.max_scanpath_length]
            if untruncated:
                candidates = untruncated
                stats["groups_preferred_not_truncated"] += 1

        row = choose_row(candidates, args.strategy, rng)

        if canonical_answer is not None:
            row["participant_answer"] = extract_answer(row)
            row["answer"] = canonical_answer
            if isinstance(row.get("messages"), list):
                for msg in reversed(row["messages"]):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        msg["content"] = canonical_answer
                        break

        selected.append(row)

    write_jsonl(args.output_jsonl, selected)
    lengths = [int(r.get("scanpath_length", 0)) for r in selected]
    stats.update({
        "output_rows": len(selected),
        "strategy": args.strategy,
        "seed": args.seed,
        "selected_length_min": min(lengths) if lengths else None,
        "selected_length_max": max(lengths) if lengths else None,
        "selected_length_mean": (sum(lengths) / len(lengths)) if lengths else None,
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "qa_jsonl": str(args.qa_jsonl) if args.qa_jsonl else None,
    })

    stats_path = args.stats_json or args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"Wrote: {args.output_jsonl}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
