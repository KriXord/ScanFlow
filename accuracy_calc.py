from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from prompt_templates import normalize_chartqapro_question_type


DEFAULT_RESULTS_DIR = Path("eval_outputs")
DEFAULT_DATASET_NAMES = ("ChartQA", )
DEFAULT_EXTRA_DATASET_SPLITS = ("ChartQA:val",)
# DEFAULT_EXTRA_DATASET_SPLITS = ("ChartBench:test",)
# DEFAULT_DATASET_NAMES = ( "ChartBench",)
# DEFAULT_EXTRA_DATASET_SPLITS = ("ChartQAPro:test",)
# DEFAULT_DATASET_NAMES = ( "ChartQAPro",)
CHARTQAPRO_QUESTION_TYPES = (
    "factoid",
    "multi choice",
    "hypothetical",
    "fact checking",
    "conversational",
)
NUMERIC_PATTERN = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_prediction_path(results_dir: Path, dataset_name: str, split: str) -> Path:
    normalized = dataset_name.strip().lower().replace("-", "_")
    return results_dir / f"{normalized}_{split}_deepseek_ocr2_predictions.jsonl"


def canonical_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower().replace("-", "").replace("_", "")
    mapping = {
        "chartqa": "ChartQA",
        "chartqapro": "ChartQAPro",
        "chartbench": "ChartBench",
        "salchartqa": "SalChartQA",
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Expected one of: {', '.join(DEFAULT_DATASET_NAMES)}."
        )
    return mapping[normalized]


def parse_dataset_split_spec(spec: str) -> tuple[str, str]:
    dataset_name, separator, split = spec.partition(":")
    dataset_name = dataset_name.strip()
    split = split.strip()
    if not separator or not dataset_name or not split:
        raise ValueError(
            f"Invalid dataset split spec '{spec}'. Expected the format 'Dataset:split', "
            "for example 'ChartQA:val'."
        )
    return canonical_dataset_name(dataset_name), split


def build_eval_jobs(
    dataset_names: list[str],
    default_split: str,
    extra_dataset_splits: list[str],
) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for dataset_name in dataset_names:
        job = (canonical_dataset_name(dataset_name), default_split)
        if job not in seen:
            jobs.append(job)
            seen.add(job)

    for spec in extra_dataset_splits:
        job = parse_dataset_split_spec(spec)
        if job not in seen:
            jobs.append(job)
            seen.add(job)

    return jobs


def build_summary_key(
    dataset_name: str,
    split: str,
    eval_jobs: list[tuple[str, str]],
) -> str:
    same_dataset_job_count = sum(1 for current_dataset, _ in eval_jobs if current_dataset == dataset_name)
    if same_dataset_job_count > 1 or split != "test":
        return f"{dataset_name}:{split}"
    return dataset_name


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def safe_predictions(record: dict[str, Any]) -> list[str]:
    predictions = record.get("predictions")
    if not isinstance(predictions, list):
        return []
    return [str(prediction or "").strip() for prediction in predictions]


def is_numeric_answer(text: Any) -> bool:
    normalized = str(text or "").strip().replace(",", "")
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", normalized))


def parse_first_number(text: Any) -> float | None:
    match = NUMERIC_PATTERN.search(str(text or ""))
    if match is None:
        return None
    numeric_text = match.group(0).replace(",", "").rstrip("%")
    try:
        return float(numeric_text)
    except ValueError:
        return None


def numeric_match_within_5_percent(prediction: Any, answer: Any) -> bool:
    pred_value = parse_first_number(prediction)
    gold_value = parse_first_number(answer)
    if pred_value is None or gold_value is None:
        return False

    if math.isclose(gold_value, 0.0, abs_tol=1e-12):
        return math.isclose(pred_value, 0.0, abs_tol=1e-12)

    relative_error = abs(pred_value - gold_value) / abs(gold_value)
    return relative_error <= 0.05


def exact_match(prediction: Any, answer: Any) -> bool:
    return normalize_text(prediction) == normalize_text(answer)


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    if len(left) < len(right):
        left, right = right, left

    previous_row = list(range(len(right) + 1))
    for left_idx, left_char in enumerate(left, start=1):
        current_row = [left_idx]
        for right_idx, right_char in enumerate(right, start=1):
            insert_cost = current_row[right_idx - 1] + 1
            delete_cost = previous_row[right_idx] + 1
            replace_cost = previous_row[right_idx - 1] + (left_char != right_char)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def anls_score(prediction: Any, answer: Any, threshold: float = 0.5) -> float:
    pred_text = normalize_text(prediction)
    gold_text = normalize_text(answer)
    if not pred_text and not gold_text:
        return 1.0
    if not pred_text or not gold_text:
        return 0.0

    distance = levenshtein_distance(pred_text, gold_text)
    normalized_distance = distance / max(len(pred_text), len(gold_text))
    score = 1.0 - normalized_distance
    return score if score >= threshold else 0.0


def update_breakdown(
    breakdown: dict[str, dict[str, float]],
    category: str,
    *,
    score: float,
    hit: bool,
) -> None:
    bucket = breakdown.setdefault(
        category,
        {
            "count": 0,
            "score_sum": 0.0,
            "hit_count": 0,
        },
    )
    bucket["count"] += 1
    bucket["score_sum"] += score
    bucket["hit_count"] += int(hit)


def finalize_breakdown(breakdown: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    finalized: dict[str, dict[str, float]] = {}
    for category, values in breakdown.items():
        count = int(values["count"])
        score_sum = float(values["score_sum"])
        hit_count = int(values["hit_count"])
        finalized[category] = {
            "count": count,
            "avg_score": score_sum / count if count else 0.0,
            "hit_accuracy": hit_count / count if count else 0.0,
        }
    return finalized


def initialize_breakdown(categories: tuple[str, ...]) -> dict[str, dict[str, float]]:
    return {
        category: {
            "count": 0,
            "score_sum": 0.0,
            "hit_count": 0,
        }
        for category in categories
    }


def score_chartqapro_answer(
    question_type: str,
    prediction: Any,
    answer: Any,
    *,
    year_flag: bool,
) -> float:
    if year_flag:
        return 1.0 if exact_match(prediction, answer) else 0.0
    if question_type == "multi choice":
        return 1.0 if exact_match(prediction, answer) else 0.0
    if question_type == "fact checking":
        return 1.0 if exact_match(prediction, answer) else 0.0
    if is_numeric_answer(answer):
        return 1.0 if numeric_match_within_5_percent(prediction, answer) else 0.0
    return anls_score(prediction, answer, threshold=0.5)


def evaluate_chartqa(path: Path, *, split: str) -> dict[str, Any]:
    records = load_jsonl(path)
    total_questions = 0
    correct_questions = 0
    missing_predictions = 0
    breakdown: dict[str, dict[str, float]] = {}

    for record in records:
        answers = [str(answer) for answer in record.get("answers", [])]
        predictions = safe_predictions(record)
        for idx, answer in enumerate(answers):
            prediction = predictions[idx] if idx < len(predictions) else ""
            if idx >= len(predictions):
                missing_predictions += 1

            if is_numeric_answer(answer):
                is_correct = numeric_match_within_5_percent(prediction, answer)
                category = "numeric"
            else:
                is_correct = exact_match(prediction, answer)
                category = "non_numeric_exact"

            total_questions += 1
            correct_questions += int(is_correct)
            update_breakdown(breakdown, category, score=float(is_correct), hit=is_correct)

    accuracy = correct_questions / total_questions if total_questions else 0.0
    return {
        "dataset": "ChartQA",
        "split": split,
        "metric_name": "accuracy",
        "score": accuracy,
        "total_questions": total_questions,
        "correct_questions": correct_questions,
        "missing_predictions": missing_predictions,
        "breakdown": finalize_breakdown(breakdown),
    }


def evaluate_chartqapro(path: Path, *, split: str) -> dict[str, Any]:
    records = load_jsonl(path)
    total_questions = 0
    score_sum = 0.0
    hit_count = 0
    missing_predictions = 0
    breakdown = initialize_breakdown(CHARTQAPRO_QUESTION_TYPES)

    for record in records:
        answers = [str(answer) for answer in record.get("answers", [])]
        predictions = safe_predictions(record)
        metadata = record.get("metadata", {}) or {}
        question_type = normalize_chartqapro_question_type(metadata.get("question_type"))
        year_flags = metadata.get("year") or []
        category = question_type if question_type in CHARTQAPRO_QUESTION_TYPES else (question_type or "unknown")

        for idx, answer in enumerate(answers):
            prediction = predictions[idx] if idx < len(predictions) else ""
            if idx >= len(predictions):
                missing_predictions += 1

            year_flag = idx < len(year_flags) and normalize_text(year_flags[idx]) == "yes"
            score = score_chartqapro_answer(
                question_type,
                prediction,
                answer,
                year_flag=year_flag,
            )

            hit = score > 0.0
            total_questions += 1
            score_sum += score
            hit_count += int(hit)
            update_breakdown(breakdown, category, score=score, hit=hit)

    average_score = score_sum / total_questions if total_questions else 0.0
    threshold_hit_accuracy = hit_count / total_questions if total_questions else 0.0
    finalized_breakdown = finalize_breakdown(breakdown)
    return {
        "dataset": "ChartQAPro",
        "split": split,
        "metric_name": "mixed_numeric_exact_anls_by_question_type",
        "score": average_score,
        "threshold_hit_accuracy": threshold_hit_accuracy,
        "total_questions": total_questions,
        "missing_predictions": missing_predictions,
        "breakdown": finalized_breakdown,
        "question_type_breakdown": finalized_breakdown,
    }


def evaluate_chartbench(path: Path, *, split: str) -> dict[str, Any]:
    records = load_jsonl(path)
    total_samples = 0
    correct_samples = 0
    total_subquestions = 0
    correct_subquestions = 0
    missing_predictions = 0
    task_breakdown: dict[str, dict[str, float]] = {}

    for record in records:
        answers = [str(answer) for answer in record.get("answers", [])]
        predictions = safe_predictions(record)
        metadata = record.get("metadata", {}) or {}
        task_name = str(metadata.get("task") or "unknown")
        sample_correct = True
        total_samples += 1

        task_bucket = task_breakdown.setdefault(
            task_name,
            {
                "total_samples": 0,
                "correct_samples": 0,
                "total_subquestions": 0,
                "correct_subquestions": 0,
                "missing_predictions": 0,
            },
        )
        task_bucket["total_samples"] += 1

        if len(predictions) < len(answers):
            missing_count = len(answers) - len(predictions)
            missing_predictions += missing_count
            task_bucket["missing_predictions"] += missing_count
            sample_correct = False

        for idx, answer in enumerate(answers):
            prediction = predictions[idx] if idx < len(predictions) else ""
            is_correct = exact_match(prediction, answer)
            total_subquestions += 1
            correct_subquestions += int(is_correct)
            task_bucket["total_subquestions"] += 1
            task_bucket["correct_subquestions"] += int(is_correct)
            if not is_correct:
                sample_correct = False

        if sample_correct and len(answers) == len(predictions):
            correct_samples += 1
            task_bucket["correct_samples"] += 1

    accuracy = correct_samples / total_samples if total_samples else 0.0
    subquestion_accuracy = correct_subquestions / total_subquestions if total_subquestions else 0.0
    finalized_task_breakdown: dict[str, dict[str, float]] = {}
    for task_name, bucket in sorted(task_breakdown.items()):
        task_total_samples = int(bucket["total_samples"])
        task_correct_samples = int(bucket["correct_samples"])
        task_total_subquestions = int(bucket["total_subquestions"])
        task_correct_subquestions = int(bucket["correct_subquestions"])
        finalized_task_breakdown[task_name] = {
            "total_samples": task_total_samples,
            "correct_samples": task_correct_samples,
            "sample_accuracy": (
                task_correct_samples / task_total_samples if task_total_samples else 0.0
            ),
            "total_subquestions": task_total_subquestions,
            "correct_subquestions": task_correct_subquestions,
            "subquestion_accuracy": (
                task_correct_subquestions / task_total_subquestions if task_total_subquestions else 0.0
            ),
            "missing_predictions": int(bucket["missing_predictions"]),
        }

    return {
        "dataset": "ChartBench",
        "split": split,
        "metric_name": "sample_accuracy_all_subitems_exact",
        "score": accuracy,
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "missing_predictions": missing_predictions,
        "total_subquestions": total_subquestions,
        "subquestion_accuracy": subquestion_accuracy,
        "task_breakdown": finalized_task_breakdown,
    }


def evaluate_dataset(dataset_name: str, split: str, path: Path) -> dict[str, Any]:
    normalized = dataset_name.strip().lower()
    if normalized == "chartqa":
        return evaluate_chartqa(path, split=split)
    if normalized == "salchartqa":
        result = evaluate_chartqa(path, split=split)
        result["dataset"] = "SalChartQA"
        return result
    if normalized == "chartqapro":
        return evaluate_chartqapro(path, split=split)
    if normalized == "chartbench":
        return evaluate_chartbench(path, split=split)
    raise ValueError(f"Unsupported dataset '{dataset_name}'.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ChartQA / ChartQAPro / ChartBench prediction JSONL files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing prediction JSONL files. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASET_NAMES),
        help="Datasets to evaluate. Default: ChartQA ChartQAPro ChartBench.",
    )
    parser.add_argument("--split", default="test", help="Split to evaluate for --datasets. Default: test.")
    parser.add_argument(
        "--extra-dataset-splits",
        nargs="*",
        default=list(DEFAULT_EXTRA_DATASET_SPLITS),
        help=(
            "Optional extra evaluation jobs in the form 'Dataset:split'. "
            "Example: --extra-dataset-splits ChartQA:val. "
            "Default includes ChartQA:val."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the evaluation summary as JSON.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary: dict[str, Any] = {}
    try:
        eval_jobs = build_eval_jobs(args.datasets, args.split, args.extra_dataset_splits)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    for dataset_name, split_name in eval_jobs:
        prediction_path = build_prediction_path(args.results_dir, dataset_name, split_name)
        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Prediction file not found for {dataset_name} ({split_name}): {prediction_path}"
            )
        summary_key = build_summary_key(dataset_name, split_name, eval_jobs)
        summary[summary_key] = evaluate_dataset(dataset_name, split_name, prediction_path)

    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
