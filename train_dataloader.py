from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Sequence

try:
    from torch.utils.data import DataLoader, IterableDataset, get_worker_info
except ImportError:  # pragma: no cover - optional dependency for training only
    DataLoader = None  # type: ignore[assignment]
    IterableDataset = object  # type: ignore[assignment]

    def get_worker_info():  # type: ignore[no-redef]
        return None

try:
    from dataloader import DEFAULT_DATASETS_ROOT, ChartSample, iter_dataset
    from prompt_templates import (
        DEFAULT_CHARTBENCH_STATEMENT_PROMPT_TEMPLATE,
        DEFAULT_CHARTBENCH_VALUE_PROMPT_TEMPLATE,
        DEFAULT_PROMPT_TEMPLATE,
        is_chartbench_value_question,
    )
except ImportError:
    from dataloader import DEFAULT_DATASETS_ROOT, ChartSample, iter_dataset  # type: ignore
    from prompt_templates import (  # type: ignore
        DEFAULT_CHARTBENCH_STATEMENT_PROMPT_TEMPLATE,
        DEFAULT_CHARTBENCH_VALUE_PROMPT_TEMPLATE,
        DEFAULT_PROMPT_TEMPLATE,
        is_chartbench_value_question,
    )


SUPPORTED_TRAIN_DATASETS = ("ChartBench", "ChartQA", "SalChartQA")
ALL_TRAIN_DATASET_ALIASES = {
    "all",
    "both",
    "mix",
    "chartbench+chartqa",
    "chartqa+chartbench",
    "chartqa+salchartqa",
    "salchartqa+chartqa",
    "chartbench+chartqa+salchartqa",
}
SUPPORTED_MULTI_DATASET_MODES = ("round_robin", "concat")


@dataclass(slots=True)
class ChartTrainExample:
    dataset: str
    split: str
    sample_id: int | str
    question_idx: int
    question: str
    answer: str
    accepted_answers: list[str] = field(default_factory=list)
    prompt: str = ""
    image_path: str | None = None
    image_bytes: bytes | None = None
    fixation_map_path: str | None = None
    scanpath_length: int | None = None
    scanpath_csv_path: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None

    @property
    def qa_id(self) -> str:
        return f"{self.dataset}:{self.sample_id}:{self.question_idx}"

    @property
    def source_qa_id(self) -> str:
        return f"{self.sample_id}:{self.question_idx}"


def _normalize_train_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower().replace("-", "").replace("_", "")
    if normalized == "chartbench":
        return "ChartBench"
    if normalized == "chartqa":
        return "ChartQA"
    if normalized in {"salchartqa", "salchartqascanpath"}:
        return "SalChartQA"
    raise ValueError(
        f"Unsupported training dataset '{dataset_name}'. Expected one of: {', '.join(SUPPORTED_TRAIN_DATASETS)}."
    )


def _normalize_train_dataset_names(dataset_name: str | Sequence[str]) -> list[str]:
    if isinstance(dataset_name, str):
        normalized = dataset_name.strip().lower().replace("_", "").replace("-", "")
        if normalized in ALL_TRAIN_DATASET_ALIASES:
            return list(SUPPORTED_TRAIN_DATASETS)
        return [_normalize_train_dataset_name(dataset_name)]

    normalized_names: list[str] = []
    seen: set[str] = set()
    for name in dataset_name:
        normalized_name = _normalize_train_dataset_name(name)
        if normalized_name in seen:
            continue
        normalized_names.append(normalized_name)
        seen.add(normalized_name)

    if not normalized_names:
        raise ValueError("dataset_name must contain at least one supported dataset.")
    return normalized_names


def _validate_train_split(dataset_name: str, split: str) -> None:
    if dataset_name == "ChartBench" and split != "train":
        raise ValueError("ChartBench training loader currently expects split='train'.")
    if dataset_name == "ChartQA" and split != "train":
        raise ValueError("ChartQA training loader currently expects split='train'.")
    if dataset_name == "SalChartQA" and split not in {"train", "val", "test"}:
        raise ValueError("SalChartQA training loader expects split to be one of: train, val, test.")


def _resolve_include_image_bytes(dataset_name: str, include_image_bytes: bool | None) -> bool:
    if include_image_bytes is None:
        return dataset_name == "ChartQA"
    return include_image_bytes


def _get_prompt_for_training_example(
    *,
    dataset_name: str,
    question: str,
    prompt_template: str,
    chartbench_statement_prompt_template: str,
    chartbench_value_prompt_template: str,
) -> str:
    if dataset_name == "ChartBench":
        template = (
            chartbench_value_prompt_template
            if is_chartbench_value_question(question)
            else chartbench_statement_prompt_template
        )
    else:
        template = prompt_template

    prompt = template.format(question=question.strip())
    if "<image>" not in prompt:
        raise ValueError("Prompt template must contain '<image>'.")
    return prompt


def _get_salchartqa_prompt(sample: ChartSample, question: str) -> str:
    messages = sample.metadata.get("messages") or []
    if messages:
        first_message = messages[0]
        if isinstance(first_message, dict) and first_message.get("content"):
            prompt = str(first_message["content"]).strip()
            if "<image>" not in prompt:
                prompt = f"<image>\n{prompt}"
            return prompt
    return DEFAULT_PROMPT_TEMPLATE.format(question=question.strip())


def _iter_sample_qa_pairs(sample: ChartSample) -> Iterator[tuple[int, str, str, list[str]]]:
    if not sample.questions:
        return

    if sample.dataset == "ChartQA":
        answer = sample.answers[0] if sample.answers else ""
        yield 0, sample.questions[0], answer, list(sample.answers)
        return

    if sample.dataset == "SalChartQA":
        answer = sample.answers[0] if sample.answers else ""
        question = sample.questions[0] if sample.questions else ""
        yield 0, question, answer, list(sample.answers)
        return

    for question_idx, question in enumerate(sample.questions):
        answer = sample.answers[question_idx] if question_idx < len(sample.answers) else ""
        accepted_answers = [answer] if answer else []
        yield question_idx, question, answer, accepted_answers


def _iter_train_examples_for_dataset(
    *,
    dataset_name: str,
    split: str,
    datasets_root: str | Path,
    include_image_bytes: bool | None,
    include_raw: bool,
    parquet_batch_size: int,
    prompt_template: str,
    chartbench_statement_prompt_template: str,
    chartbench_value_prompt_template: str,
) -> Iterator[ChartTrainExample]:
    _validate_train_split(dataset_name, split)
    current_include_image_bytes = _resolve_include_image_bytes(dataset_name, include_image_bytes)

    for sample in iter_dataset(
        dataset_name=dataset_name,
        split=split,
        datasets_root=datasets_root,
        include_image_bytes=current_include_image_bytes,
        include_raw=include_raw,
        batch_size=parquet_batch_size,
    ):
        for question_idx, question, answer, accepted_answers in _iter_sample_qa_pairs(sample):
            source_qa_id = f"{sample.sample_id}:{question_idx}"
            global_qa_id = f"{sample.dataset}:{source_qa_id}"
            if sample.dataset == "SalChartQA":
                prompt = _get_salchartqa_prompt(sample, question)
            else:
                prompt = _get_prompt_for_training_example(
                    dataset_name=sample.dataset,
                    question=question,
                    prompt_template=prompt_template,
                    chartbench_statement_prompt_template=chartbench_statement_prompt_template,
                    chartbench_value_prompt_template=chartbench_value_prompt_template,
                )

            yield ChartTrainExample(
                dataset=sample.dataset,
                split=sample.split,
                sample_id=sample.sample_id,
                question_idx=question_idx,
                question=question,
                answer=answer,
                accepted_answers=accepted_answers,
                prompt=prompt,
                image_path=sample.image_path,
                image_bytes=sample.image_bytes,
                fixation_map_path=sample.metadata.get("fixation_map_path"),
                scanpath_length=(
                    int(sample.metadata["scanpath_length"])
                    if sample.metadata.get("scanpath_length") is not None else None
                ),
                scanpath_csv_path=sample.metadata.get("scanpath_csv_path"),
                messages=sample.metadata.get("messages") or [],
                metadata={
                    **sample.metadata,
                    "qa_id": global_qa_id,
                    "source_qa_id": source_qa_id,
                    "source_dataset": sample.dataset,
                    "source_num_questions": len(sample.questions),
                    "source_num_answers": len(sample.answers),
                },
                raw=sample.raw,
            )


def iter_train_examples(
    dataset_name: str | Sequence[str],
    *,
    split: str = "train",
    datasets_root: str | Path = DEFAULT_DATASETS_ROOT,
    include_image_bytes: bool | None = None,
    include_raw: bool = False,
    parquet_batch_size: int = 128,
    multi_dataset_mode: str = "round_robin",
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    chartbench_statement_prompt_template: str = DEFAULT_CHARTBENCH_STATEMENT_PROMPT_TEMPLATE,
    chartbench_value_prompt_template: str = DEFAULT_CHARTBENCH_VALUE_PROMPT_TEMPLATE,
) -> Iterator[ChartTrainExample]:
    normalized_dataset_names = _normalize_train_dataset_names(dataset_name)
    if multi_dataset_mode not in SUPPORTED_MULTI_DATASET_MODES:
        raise ValueError(
            f"Unsupported multi_dataset_mode '{multi_dataset_mode}'. "
            f"Expected one of: {', '.join(SUPPORTED_MULTI_DATASET_MODES)}."
        )

    if len(normalized_dataset_names) == 1 or multi_dataset_mode == "concat":
        for normalized_dataset_name in normalized_dataset_names:
            yield from _iter_train_examples_for_dataset(
                dataset_name=normalized_dataset_name,
                split=split,
                datasets_root=datasets_root,
                include_image_bytes=include_image_bytes,
                include_raw=include_raw,
                parquet_batch_size=parquet_batch_size,
                prompt_template=prompt_template,
                chartbench_statement_prompt_template=chartbench_statement_prompt_template,
                chartbench_value_prompt_template=chartbench_value_prompt_template,
            )
        return

    iterators = {
        normalized_dataset_name: _iter_train_examples_for_dataset(
            dataset_name=normalized_dataset_name,
            split=split,
            datasets_root=datasets_root,
            include_image_bytes=include_image_bytes,
            include_raw=include_raw,
            parquet_batch_size=parquet_batch_size,
            prompt_template=prompt_template,
            chartbench_statement_prompt_template=chartbench_statement_prompt_template,
            chartbench_value_prompt_template=chartbench_value_prompt_template,
        )
        for normalized_dataset_name in normalized_dataset_names
    }
    active_names = list(normalized_dataset_names)
    while active_names:
        next_active_names: list[str] = []
        for normalized_dataset_name in active_names:
            iterator = iterators[normalized_dataset_name]
            try:
                yield next(iterator)
                next_active_names.append(normalized_dataset_name)
            except StopIteration:
                continue
        active_names = next_active_names


def load_train_examples(
    dataset_name: str | Sequence[str],
    *,
    split: str = "train",
    datasets_root: str | Path = DEFAULT_DATASETS_ROOT,
    include_image_bytes: bool | None = None,
    include_raw: bool = False,
    parquet_batch_size: int = 128,
    multi_dataset_mode: str = "round_robin",
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    chartbench_statement_prompt_template: str = DEFAULT_CHARTBENCH_STATEMENT_PROMPT_TEMPLATE,
    chartbench_value_prompt_template: str = DEFAULT_CHARTBENCH_VALUE_PROMPT_TEMPLATE,
) -> list[ChartTrainExample]:
    return list(
        iter_train_examples(
            dataset_name=dataset_name,
            split=split,
            datasets_root=datasets_root,
            include_image_bytes=include_image_bytes,
            include_raw=include_raw,
            parquet_batch_size=parquet_batch_size,
            multi_dataset_mode=multi_dataset_mode,
            prompt_template=prompt_template,
            chartbench_statement_prompt_template=chartbench_statement_prompt_template,
            chartbench_value_prompt_template=chartbench_value_prompt_template,
        )
    )


def _load_fixation_npz(path: str | Path) -> tuple[Any, Any, int]:
    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - training-only dependency
        raise RuntimeError("Loading SalChartQA fixation targets requires numpy and torch.") from exc

    data = np.load(Path(path), allow_pickle=True)
    fixation_targets = torch.from_numpy(data["fixation_targets"]).float()
    fixation_mask = torch.from_numpy(data["fixation_mask"]).bool()
    scanpath_length = int(data["scanpath_length"])

    if fixation_targets.ndim != 2:
        raise ValueError(f"Expected fixation_targets to be [T, N], got {tuple(fixation_targets.shape)} at {path}.")
    if fixation_mask.ndim != 1:
        raise ValueError(f"Expected fixation_mask to be [T], got {tuple(fixation_mask.shape)} at {path}.")
    if fixation_targets.shape[0] != fixation_mask.shape[0]:
        raise ValueError(
            "fixation_targets and fixation_mask disagree on max steps: "
            f"{tuple(fixation_targets.shape)} vs {tuple(fixation_mask.shape)} at {path}."
        )

    return fixation_targets, fixation_mask, scanpath_length


def collate_train_examples(
    batch: list[ChartTrainExample],
    *,
    load_fixation_targets: bool = False,
) -> dict[str, Any]:
    collated = {
        "dataset": [example.dataset for example in batch],
        "split": [example.split for example in batch],
        "sample_id": [example.sample_id for example in batch],
        "question_idx": [example.question_idx for example in batch],
        "qa_id": [example.qa_id for example in batch],
        "source_qa_id": [example.source_qa_id for example in batch],
        "question": [example.question for example in batch],
        "answer": [example.answer for example in batch],
        "accepted_answers": [example.accepted_answers for example in batch],
        "prompt": [example.prompt for example in batch],
        "image_path": [example.image_path for example in batch],
        "image_bytes": [example.image_bytes for example in batch],
        "fixation_map_path": [example.fixation_map_path for example in batch],
        "scanpath_length": [example.scanpath_length for example in batch],
        "scanpath_csv_path": [example.scanpath_csv_path for example in batch],
        "messages": [example.messages for example in batch],
        "metadata": [example.metadata for example in batch],
        "raw": [example.raw for example in batch],
    }

    if not load_fixation_targets:
        return collated

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training-only dependency
        raise RuntimeError("Collating SalChartQA fixation targets requires torch.") from exc

    fixation_targets = []
    fixation_masks = []
    scanpath_lengths = []
    for example in batch:
        if not example.fixation_map_path:
            raise ValueError(
                "load_fixation_targets=True requires every batch example to have fixation_map_path. "
                f"Missing for {example.qa_id}."
            )

        current_targets, current_mask, npz_scanpath_length = _load_fixation_npz(example.fixation_map_path)
        if example.scanpath_length is not None and int(example.scanpath_length) != npz_scanpath_length:
            raise ValueError(
                f"scanpath_length mismatch for {example.qa_id}: JSONL has {example.scanpath_length}, "
                f"NPZ has {npz_scanpath_length}."
            )

        fixation_targets.append(current_targets)
        fixation_masks.append(current_mask)
        scanpath_lengths.append(
            torch.tensor(
                int(example.scanpath_length if example.scanpath_length is not None else npz_scanpath_length),
                dtype=torch.long,
            )
        )

    collated["fixation_targets"] = torch.stack(fixation_targets, dim=0)
    collated["fixation_mask"] = torch.stack(fixation_masks, dim=0)
    collated["scanpath_lengths"] = torch.stack(scanpath_lengths, dim=0)
    return collated


class ChartTrainIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str | Sequence[str],
        *,
        split: str = "train",
        datasets_root: str | Path = DEFAULT_DATASETS_ROOT,
        include_image_bytes: bool | None = None,
        include_raw: bool = False,
        parquet_batch_size: int = 128,
        multi_dataset_mode: str = "round_robin",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        chartbench_statement_prompt_template: str = DEFAULT_CHARTBENCH_STATEMENT_PROMPT_TEMPLATE,
        chartbench_value_prompt_template: str = DEFAULT_CHARTBENCH_VALUE_PROMPT_TEMPLATE,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.datasets_root = Path(datasets_root)
        self.include_image_bytes = include_image_bytes
        self.include_raw = include_raw
        self.parquet_batch_size = parquet_batch_size
        self.multi_dataset_mode = multi_dataset_mode
        self.prompt_template = prompt_template
        self.chartbench_statement_prompt_template = chartbench_statement_prompt_template
        self.chartbench_value_prompt_template = chartbench_value_prompt_template

    def __iter__(self) -> Iterator[ChartTrainExample]:
        worker_info = get_worker_info()
        for example_idx, example in enumerate(
            iter_train_examples(
                dataset_name=self.dataset_name,
                split=self.split,
                datasets_root=self.datasets_root,
                include_image_bytes=self.include_image_bytes,
                include_raw=self.include_raw,
                parquet_batch_size=self.parquet_batch_size,
                multi_dataset_mode=self.multi_dataset_mode,
                prompt_template=self.prompt_template,
                chartbench_statement_prompt_template=self.chartbench_statement_prompt_template,
                chartbench_value_prompt_template=self.chartbench_value_prompt_template,
            )
        ):
            if worker_info is None or example_idx % worker_info.num_workers == worker_info.id:
                yield example


def build_train_dataloader(
    dataset_name: str | Sequence[str],
    *,
    split: str = "train",
    datasets_root: str | Path = DEFAULT_DATASETS_ROOT,
    dataloader_batch_size: int = 8,
    num_workers: int = 0,
    load_fixation_targets: bool = False,
    include_image_bytes: bool | None = None,
    include_raw: bool = False,
    parquet_batch_size: int = 128,
    multi_dataset_mode: str = "round_robin",
    pin_memory: bool = False,
    drop_last: bool = False,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    chartbench_statement_prompt_template: str = DEFAULT_CHARTBENCH_STATEMENT_PROMPT_TEMPLATE,
    chartbench_value_prompt_template: str = DEFAULT_CHARTBENCH_VALUE_PROMPT_TEMPLATE,
):
    if DataLoader is None:
        raise RuntimeError("Building a training DataLoader requires torch to be installed.")
    if dataloader_batch_size <= 0:
        raise ValueError("dataloader_batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    dataset = ChartTrainIterableDataset(
        dataset_name=dataset_name,
        split=split,
        datasets_root=datasets_root,
        include_image_bytes=include_image_bytes,
        include_raw=include_raw,
        parquet_batch_size=parquet_batch_size,
        multi_dataset_mode=multi_dataset_mode,
        prompt_template=prompt_template,
        chartbench_statement_prompt_template=chartbench_statement_prompt_template,
        chartbench_value_prompt_template=chartbench_value_prompt_template,
    )
    return DataLoader(
        dataset,
        batch_size=dataloader_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=partial(
            collate_train_examples,
            load_fixation_targets=load_fixation_targets,
        ),
    )



# from train_dataloader import build_train_dataloader
if __name__ == "__main__":

    chartbench_loader = build_train_dataloader(
        "ChartBench",
        datasets_root="datasets",
        dataloader_batch_size=8,
    )

    chartqa_loader = build_train_dataloader(
        "ChartQA",
        datasets_root="datasets",
        dataloader_batch_size=8,
    )

    mixed_loader = build_train_dataloader(
        ["ChartBench", "ChartQA"],
        datasets_root="datasets",
        dataloader_batch_size=8,
        multi_dataset_mode="round_robin",
    )

    batch = next(iter(mixed_loader))
    print(batch["dataset"])
    print(batch["qa_id"])
    print(batch["question"])
    print(batch["answer"])

    salchartqa_loader = build_train_dataloader(
        "SalChartQA",
        datasets_root="datasets",
        dataloader_batch_size=2,
        load_fixation_targets=True,
    )
    salchartqa_batch = next(iter(salchartqa_loader))
    print(salchartqa_batch["dataset"])
    print(salchartqa_batch["qa_id"])
    print(salchartqa_batch["fixation_targets"].shape)
    print(salchartqa_batch["fixation_mask"].shape)
    print(salchartqa_batch["scanpath_lengths"])
