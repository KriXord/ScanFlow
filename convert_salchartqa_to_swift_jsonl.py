from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def resolve_path(path_value: str | None, root: Path, absolute_paths: bool) -> str | None:
    if not path_value:
        return None

    path = Path(path_value)
    if not path.is_absolute():
        path = root / path

    if absolute_paths:
        return str(path.resolve())

    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def convert_record(record: dict[str, Any], root: Path, absolute_paths: bool, keep_metadata: bool) -> dict[str, Any]:
    image_path = resolve_path(record.get("image_path"), root, absolute_paths)
    fixation_map_path = resolve_path(record.get("fixation_map_path"), root, absolute_paths)

    if image_path is None:
        raise ValueError(f"Missing image_path for sample_id={record.get('sample_id')}")
    if fixation_map_path is None:
        raise ValueError(f"Missing fixation_map_path for sample_id={record.get('sample_id')}")

    messages = record.get("messages")
    if not messages:
        question = str(record.get("question", "")).strip()
        answer = str(record.get("answer", "")).strip()
        messages = [
            {"role": "user", "content": f"<image>\n{question}"},
            {"role": "assistant", "content": answer},
        ]

    converted = {
        "messages": messages,
        "images": [image_path],
        "fixation_map_path": fixation_map_path,
        "scanpath_length": int(record["scanpath_length"]),
    }

    if keep_metadata:
        for key in [
            "sample_id",
            "image_id",
            "image_name",
            "question_id",
            "question",
            "answer",
            "participant_id",
            "scanpath_csv_path",
            "raw_image_size",
            "target_type",
            "qa_match_type",
        ]:
            if key in record:
                converted[key] = record[key]

    return converted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert SalChartQA scanpath JSONL rows into ms-swift-friendly JSONL rows."
    )
    parser.add_argument("--input", type=Path, required=True, help="Original SalChartQA scanpath JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Converted JSONL output path.")
    parser.add_argument(
        "--salchartqa-root",
        type=Path,
        required=True,
        help="Root containing raw_img/, fixation_maps/, and the source JSONL.",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Keep image/fixation paths relative to --salchartqa-root instead of writing absolute paths.",
    )
    parser.add_argument(
        "--drop-metadata",
        action="store_true",
        help="Only write messages/images/fixation_map_path/scanpath_length.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = args.salchartqa_root.expanduser().resolve()
    input_path = args.input.expanduser()
    output_path = args.output.expanduser()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line_idx, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            if args.limit is not None and total >= args.limit:
                break

            record = json.loads(line)
            converted = convert_record(
                record,
                root=root,
                absolute_paths=not args.relative_paths,
                keep_metadata=not args.drop_metadata,
            )
            f_out.write(json.dumps(converted, ensure_ascii=False) + "\n")
            total += 1

    print(f"Wrote {total} converted samples to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
