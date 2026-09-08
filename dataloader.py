from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence


DEFAULT_DATASETS_ROOT = Path("datasets")


class DatasetLoaderError(RuntimeError):
    """Raised when a dataset cannot be loaded with the local environment."""


@dataclass(slots=True)
class ChartSample:
    """A normalized sample shape shared by the three chart datasets."""

    dataset: str
    split: str
    sample_id: int | str
    questions: list[str]
    answers: list[str]
    image_path: str | None = None
    image_bytes: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None

    @property
    def question(self) -> str | None:
        return self.questions[0] if self.questions else None

    @property
    def answer(self) -> str | None:
        return self.answers[0] if self.answers else None


def _normalize_dataset_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "chartbench": "chartbench",
        "chartqa": "chartqa",
        "chartqapro": "chartqapro",
        "salchartqa": "salchartqa",
        "salchartqascanpath": "salchartqa",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported dataset '{name}'. Expected one of: ChartBench, ChartQA, ChartQAPro, SalChartQA."
        )
    return aliases[normalized]


def _resolve_image_path(dataset_root: Path, image_value: str | None) -> str | None:
    if not image_value:
        return None
    image_path = Path(image_value)
    if image_path.is_absolute():
        return str(image_path)
    if image_value.startswith("./"):
        image_value = image_value[2:]
    return str((dataset_root / image_value).resolve())


def _ensure_exists(path: Path, what: str) -> Path:
    if not path.exists():
        raise DatasetLoaderError(f"{what} does not exist: {path}")
    return path


def _read_bytes(path: str | None) -> bytes | None:
    if path is None:
        return None
    return Path(path).read_bytes()


def _resolve_existing_file(dataset_root: Path, candidates: Sequence[str]) -> Path:
    for candidate in candidates:
        path = dataset_root / candidate
        if path.exists():
            return path
    raise DatasetLoaderError(
        "Could not find any candidate annotation file under "
        f"{dataset_root}: {', '.join(candidates)}"
    )


def _get_first_image_value(raw: dict[str, Any]) -> str | None:
    image_value = raw.get("image_path")
    if image_value:
        return str(image_value)

    images = raw.get("images")
    if isinstance(images, str):
        return images
    if isinstance(images, Sequence) and images:
        first_image = images[0]
        return str(first_image) if first_image else None
    return None


def _extract_salchartqa_question_answer(raw: dict[str, Any]) -> tuple[str, str]:
    question = str(raw.get("question", "") or "").strip()
    answer = str(raw.get("answer", "") or "").strip()

    messages = raw.get("messages") or []
    if not question:
        for message in messages:
            if message.get("role") == "user":
                content = str(message.get("content", "") or "")
                question = content.replace("<image>", "").strip()
                break
    if not answer:
        for message in messages:
            if message.get("role") == "assistant":
                answer = str(message.get("content", "") or "").strip()
                break
    return question, answer


def _infer_split_from_annotation_path(path: Path, requested_split: str) -> str:
    name = path.name.lower()
    for split_name in ("train", "val", "test"):
        if split_name in name:
            return split_name
    return requested_split


def _require_duckdb():
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise DatasetLoaderError(
            "Loading ChartQA/ChartQAPro requires the 'duckdb' Python package."
        ) from exc
    return duckdb


def _iter_duckdb_rows(parquet_paths: Sequence[Path], batch_size: int) -> Iterator[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    duckdb = _require_duckdb()
    relation = duckdb.read_parquet([str(path) for path in parquet_paths])
    columns = list(relation.columns)
    while True:
        rows = relation.fetchmany(batch_size)
        if not rows:
            break
        for row in rows:
            yield dict(zip(columns, row))


def iter_chartbench(
    split: str = "test",
    dataset_root: str | Path = DEFAULT_DATASETS_ROOT / "ChartBench",
    *,
    include_raw: bool = False,
) -> Iterator[ChartSample]:
    """
    Iterate over ChartBench samples.

    Notes:
    - ChartBench stores annotations in JSONL.
    - Image paths are relative paths inside the dataset directory.
    - If you only downloaded the zip files and have not extracted them, image_path
      may point to files that do not exist yet.
    """

    split_to_file = {
        "train": "train.jsonl",
        "train_data": "train.jsonl",
        "test": "test.jsonl",
        "test_data": "test.jsonl",
    }
    if split not in split_to_file:
        raise ValueError("ChartBench split must be one of: train, train_data, test, test_data.")

    dataset_root = Path(dataset_root)
    annotation_path = _ensure_exists(dataset_root / split_to_file[split], "ChartBench annotation file")

    with annotation_path.open("r", encoding="utf-8") as handle:
        for fallback_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            conversation = raw.get("conversation") or []
            sample = ChartSample(
                dataset="ChartBench",
                split="train" if split.startswith("train") else "test",
                sample_id=raw.get("id", fallback_idx),
                questions=[turn.get("query", "") for turn in conversation if turn.get("query")],
                answers=[str(turn.get("label", "")) for turn in conversation if "label" in turn],
                image_path=_resolve_image_path(dataset_root, raw.get("image")),
                metadata={
                    "chart_type": raw.get("type", {}).get("chart"),
                    "image_type": raw.get("type", {}).get("image"),
                    "task": raw.get("type", {}).get("task"),
                    "qa_metric": raw.get("type", {}).get("QA"),
                    "num_turns": len(conversation),
                },
                raw=raw if include_raw else None,
            )
            yield sample


def load_chartbench(
    split: str = "test",
    dataset_root: str | Path = DEFAULT_DATASETS_ROOT / "ChartBench",
    *,
    include_raw: bool = False,
) -> list[ChartSample]:
    return list(iter_chartbench(split=split, dataset_root=dataset_root, include_raw=include_raw))


def iter_chartqa(
    split: str = "test",
    dataset_root: str | Path = DEFAULT_DATASETS_ROOT / "ChartQA",
    *,
    include_image_bytes: bool = True,
    include_raw: bool = False,
    batch_size: int = 128,
) -> Iterator[ChartSample]:
    """Iterate over ChartQA samples stored in parquet shards."""

    split_to_glob = {
        "train": "train-*.parquet",
        "val": "val-*.parquet",
        "test": "test-*.parquet",
    }
    if split not in split_to_glob:
        raise ValueError("ChartQA split must be one of: train, val, test.")

    dataset_root = Path(dataset_root)
    parquet_paths = sorted((dataset_root / "data").glob(split_to_glob[split]))
    if not parquet_paths:
        raise DatasetLoaderError(f"No ChartQA parquet files found for split '{split}' under {dataset_root}.")

    human_or_machine_map = {0: "human", 1: "machine"}
    for row_index, raw in enumerate(_iter_duckdb_rows(parquet_paths, batch_size=batch_size)):
        image = raw.get("image") or {}
        label = raw.get("label") or []
        sample = ChartSample(
            dataset="ChartQA",
            split=split,
            sample_id=row_index,
            questions=[raw["query"]] if raw.get("query") else [],
            answers=[str(answer) for answer in label],
            image_path=image.get("path"),
            image_bytes=image.get("bytes") if include_image_bytes else None,
            metadata={
                "human_or_machine": human_or_machine_map.get(raw.get("human_or_machine")),
                "human_or_machine_id": raw.get("human_or_machine"),
            },
            raw=raw if include_raw else None,
        )
        yield sample


def load_chartqa(
    split: str = "test",
    dataset_root: str | Path = DEFAULT_DATASETS_ROOT / "ChartQA",
    *,
    include_image_bytes: bool = True,
    include_raw: bool = False,
    batch_size: int = 128,
) -> list[ChartSample]:
    return list(
        iter_chartqa(
            split=split,
            dataset_root=dataset_root,
            include_image_bytes=include_image_bytes,
            include_raw=include_raw,
            batch_size=batch_size,
        )
    )


def iter_chartqapro(
    split: str = "test",
    dataset_root: str | Path = DEFAULT_DATASETS_ROOT / "ChartQAPro",
    *,
    include_image_bytes: bool = True,
    include_raw: bool = False,
    batch_size: int = 128,
) -> Iterator[ChartSample]:
    """Iterate over ChartQAPro samples stored in parquet shards."""

    if split != "test":
        raise ValueError("ChartQAPro currently only provides the 'test' split.")

    dataset_root = Path(dataset_root)
    parquet_paths = sorted((dataset_root / "data").glob("test-*.parquet"))
    if not parquet_paths:
        raise DatasetLoaderError(f"No ChartQAPro parquet files found under {dataset_root}.")

    for row_index, raw in enumerate(_iter_duckdb_rows(parquet_paths, batch_size=batch_size)):
        questions = [str(question) for question in (raw.get("Question") or [])]
        answers = [str(answer) for answer in (raw.get("Answer") or [])]
        years = [str(year) for year in (raw.get("Year") or [])]
        sample = ChartSample(
            dataset="ChartQAPro",
            split="test",
            sample_id=row_index,
            questions=questions,
            answers=answers,
            image_bytes=raw.get("image") if include_image_bytes else None,
            metadata={
                "question_type": raw.get("Question Type"),
                "year": years,
                "paragraph": raw.get("Paragraph"),
            },
            raw=raw if include_raw else None,
        )
        yield sample


def load_chartqapro(
    split: str = "test",
    dataset_root: str | Path = DEFAULT_DATASETS_ROOT / "ChartQAPro",
    *,
    include_image_bytes: bool = True,
    include_raw: bool = False,
    batch_size: int = 128,
) -> list[ChartSample]:
    return list(
        iter_chartqapro(
            split=split,
            dataset_root=dataset_root,
            include_image_bytes=include_image_bytes,
            include_raw=include_raw,
            batch_size=batch_size,
        )
    )


def iter_salchartqa(
    split: str = "train",
    dataset_root: str | Path = DEFAULT_DATASETS_ROOT / "SalChartQA",
    *,
    include_image_bytes: bool = False,
    include_raw: bool = False,
    annotation_file: str | Path | None = None,
) -> Iterator[ChartSample]:
    """Iterate over scanpath-supervised SalChartQA JSONL samples.

    The expected JSONL is the file produced by the SalChartQA join script:
    one row per participant scanpath, with fields such as image_path,
    fixation_map_path, scanpath_length, messages, question, and answer.
    """

    split_to_candidates = {
        "train": (
            "splits/salchartqa_scanpath_train_correct.jsonl",
            "splits/salchartqa_scanpath_train_correct_swift.jsonl",
            "splits/salchartqa_qa_train.jsonl",
            "salchartqa_scanpath_train_correct.jsonl",
            "salchartqa_scanpath_train_correct_swift.jsonl",
            "salchartqa_qa_train.jsonl",
            "salchartqa_scanpath_train.jsonl",
            "train.jsonl",
        ),
        "val": (
            "splits/salchartqa_qa_val.jsonl",
            "splits/salchartqa_scanpath_val_correct.jsonl",
            "splits/salchartqa_scanpath_val.jsonl",
            "salchartqa_scanpath_val_correct.jsonl",
            "salchartqa_scanpath_val.jsonl",
            "salchartqa_qa_val.jsonl",
            "val.jsonl",
        ),
        "test": (
            "splits/salchartqa_qa_test.jsonl",
            "splits/salchartqa_scanpath_test_correct.jsonl",
            "splits/salchartqa_scanpath_test.jsonl",
            "salchartqa_scanpath_test_correct.jsonl",
            "salchartqa_scanpath_test.jsonl",
            "salchartqa_qa_test.jsonl",
            "test.jsonl",
        ),
    }
    if split not in split_to_candidates:
        raise ValueError("SalChartQA split must be one of: train, val, test.")

    dataset_root = Path(dataset_root)
    if annotation_file is None:
        annotation_path = _resolve_existing_file(dataset_root, split_to_candidates[split])
    else:
        annotation_path = Path(annotation_file)
        if not annotation_path.is_absolute():
            annotation_path = dataset_root / annotation_path
        annotation_path = _ensure_exists(annotation_path, "SalChartQA annotation file")
    actual_split = _infer_split_from_annotation_path(annotation_path, split)

    with annotation_path.open("r", encoding="utf-8") as handle:
        for fallback_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            image_path = _resolve_image_path(dataset_root, _get_first_image_value(raw))
            fixation_map_path = _resolve_image_path(dataset_root, raw.get("fixation_map_path"))
            scanpath_csv_path = _resolve_image_path(dataset_root, raw.get("scanpath_csv_path"))
            question, answer = _extract_salchartqa_question_answer(raw)
            messages = raw.get("messages") or [
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer},
            ]

            yield ChartSample(
                dataset="SalChartQA",
                split=actual_split,
                sample_id=raw.get("sample_id", fallback_idx),
                questions=[question] if question else [],
                answers=[answer] if answer else [],
                image_path=image_path,
                image_bytes=_read_bytes(image_path) if include_image_bytes else None,
                metadata={
                    "requested_split": split,
                    "actual_split": actual_split,
                    "image_id": raw.get("image_id"),
                    "image_name": raw.get("image_name"),
                    "question_id": raw.get("question_id"),
                    "question_type": raw.get("question_type"),
                    "is_answer_numerical": raw.get("is_answer_numerical"),
                    "participant_id": raw.get("participant_id"),
                    "version": raw.get("version"),
                    "is_approved": raw.get("is_approved"),
                    "is_correct": raw.get("is_correct"),
                    "image_type": raw.get("image_type"),
                    "is_chart_simple": raw.get("is_chart_simple"),
                    "answer_label": raw.get("answer_label"),
                    "scanpath_csv_path": scanpath_csv_path,
                    "number_of_clicks": raw.get("number_of_clicks"),
                    "total_duration": raw.get("total_duration"),
                    "fixation_map_path": fixation_map_path,
                    "scanpath_length": raw.get("scanpath_length"),
                    "raw_image_size": raw.get("raw_image_size"),
                    "target_type": raw.get("target_type"),
                    "qa_match_type": raw.get("qa_match_type"),
                    "messages": messages,
                    "annotation_path": str(annotation_path),
                },
                raw=raw if include_raw else None,
            )


def load_salchartqa(
    split: str = "train",
    dataset_root: str | Path = DEFAULT_DATASETS_ROOT / "SalChartQA",
    *,
    include_image_bytes: bool = False,
    include_raw: bool = False,
    annotation_file: str | Path | None = None,
) -> list[ChartSample]:
    return list(
        iter_salchartqa(
            split=split,
            dataset_root=dataset_root,
            include_image_bytes=include_image_bytes,
            include_raw=include_raw,
            annotation_file=annotation_file,
        )
    )


def iter_dataset(
    dataset_name: str,
    split: str = "test",
    datasets_root: str | Path = DEFAULT_DATASETS_ROOT,
    *,
    include_image_bytes: bool = True,
    include_raw: bool = False,
    batch_size: int = 128,
) -> Iterator[ChartSample]:
    """
    Unified lazy-loading entrypoint.

    Example:
        samples = iter_dataset("ChartQA", split="test")
        first = next(samples)
    """

    datasets_root = Path(datasets_root)
    dataset_key = _normalize_dataset_name(dataset_name)
    if dataset_key == "chartbench":
        return iter_chartbench(
            split=split,
            dataset_root=datasets_root / "ChartBench",
            include_raw=include_raw,
        )
    if dataset_key == "chartqa":
        return iter_chartqa(
            split=split,
            dataset_root=datasets_root / "ChartQA",
            include_image_bytes=include_image_bytes,
            include_raw=include_raw,
            batch_size=batch_size,
        )
    if dataset_key == "salchartqa":
        return iter_salchartqa(
            split=split,
            dataset_root=datasets_root / "SalChartQA",
            include_image_bytes=include_image_bytes,
            include_raw=include_raw,
        )
    return iter_chartqapro(
        split=split,
        dataset_root=datasets_root / "ChartQAPro",
        include_image_bytes=include_image_bytes,
        include_raw=include_raw,
        batch_size=batch_size,
    )


def load_dataset(
    dataset_name: str,
    split: str = "test",
    datasets_root: str | Path = DEFAULT_DATASETS_ROOT,
    *,
    include_image_bytes: bool = True,
    include_raw: bool = False,
    batch_size: int = 128,
) -> list[ChartSample]:
    return list(
        iter_dataset(
            dataset_name=dataset_name,
            split=split,
            datasets_root=datasets_root,
            include_image_bytes=include_image_bytes,
            include_raw=include_raw,
            batch_size=batch_size,
        )
    )
