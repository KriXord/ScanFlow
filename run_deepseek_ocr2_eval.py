from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from evaluation.dataloader import DEFAULT_DATASETS_ROOT, ChartSample, iter_dataset
    from evaluation.prompt_templates import DEFAULT_PROMPT_TEMPLATE
except ImportError:
    from dataloader import DEFAULT_DATASETS_ROOT, ChartSample, iter_dataset  # type: ignore
    from prompt_templates import DEFAULT_PROMPT_TEMPLATE  # type: ignore


DEFAULT_DATASET_NAMES = ("ChartBench", "ChartQA", "ChartQAPro")
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"


def detect_image_suffix(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".img"


def jsonl_append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    existing: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "sample_id" in record:
                existing.add(str(record["sample_id"]))
    return existing


def build_output_path(output_dir: Path, dataset_name: str, split: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = dataset_name.strip().lower().replace("-", "_")
    return output_dir / f"{normalized}_{split}_deepseek_ocr2_predictions.jsonl"


def materialize_sample_image(sample: ChartSample, temp_dir: Path) -> tuple[str, Path | None]:
    if sample.image_path:
        image_path = Path(sample.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        return str(image_path), None

    if sample.image_bytes is None:
        raise ValueError(
            f"Sample {sample.sample_id} from {sample.dataset} has neither image_path nor image_bytes."
        )

    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = detect_image_suffix(sample.image_bytes)
    temp_path = temp_dir / f"{sample.dataset}_{sample.sample_id}_{uuid4().hex}{suffix}"
    temp_path.write_bytes(sample.image_bytes)
    return str(temp_path), temp_path


class DeepSeekOCR2Runner:
    def __init__(
        self,
        *,
        model_name: str,
        prompt_template: str,
        scratch_output_dir: Path,
        base_size: int,
        image_size: int,
        crop_mode: bool,
        attn_implementation: str,
        dtype: str,
    ) -> None:
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.scratch_output_dir = scratch_output_dir
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.attn_implementation = attn_implementation
        self.dtype_name = dtype
        self.tokenizer = None
        self.model = None
        self.torch = None

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "DeepSeek-OCR-2 evaluation requires an environment with both 'torch' and "
                "'transformers' installed."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("DeepSeek-OCR-2 inference requires CUDA, but no GPU is visible.")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self.dtype_name not in dtype_map:
            raise ValueError(f"Unsupported dtype '{self.dtype_name}'.")

        self.scratch_output_dir.mkdir(parents=True, exist_ok=True)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            _attn_implementation=self.attn_implementation,
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model = self.model.eval().cuda().to(dtype_map[self.dtype_name])

    def answer(self, *, image_file: str, question: str) -> str:
        self.load()
        prompt = self.prompt_template.format(question=question.strip())
        if "<image>" not in prompt:
            raise ValueError("prompt_template must contain the '<image>' placeholder.")

        output = self.model.infer(  # type: ignore[union-attr]
            self.tokenizer,
            prompt=prompt,
            image_file=image_file,
            output_path=str(self.scratch_output_dir),
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=self.crop_mode,
            save_results=False,
            eval_mode=True,
        )
        return output.strip()


def evaluate_dataset(
    *,
    dataset_name: str,
    split: str,
    datasets_root: Path,
    output_path: Path,
    temp_image_dir: Path,
    runner: DeepSeekOCR2Runner,
    batch_size: int,
    overwrite: bool,
    max_samples: int | None,
    log_every: int,
    continue_on_error: bool,
) -> dict[str, int]:
    if overwrite and output_path.exists():
        output_path.unlink()

    completed_ids = load_existing_sample_ids(output_path)
    processed = 0
    skipped = 0
    errors = 0

    for sample in iter_dataset(
        dataset_name=dataset_name,
        split=split,
        datasets_root=datasets_root,
        include_image_bytes=True,
        batch_size=batch_size,
    ):
        sample_key = str(sample.sample_id)
        if sample_key in completed_ids:
            skipped += 1
            continue

        if max_samples is not None and processed >= max_samples:
            break

        image_file: str | None = None
        temp_file: Path | None = None
        started_at = time.time()

        try:
            image_file, temp_file = materialize_sample_image(sample, temp_image_dir / dataset_name.lower())
            predictions = [runner.answer(image_file=image_file, question=question) for question in sample.questions]
            record = {
                "dataset": dataset_name,
                "split": split,
                "sample_id": sample.sample_id,
                "questions": sample.questions,
                "answers": sample.answers,
                "predictions": predictions,
                "metadata": sample.metadata,
                "elapsed_seconds": round(time.time() - started_at, 3),
            }
        except Exception as exc:
            errors += 1
            record = {
                "dataset": dataset_name,
                "split": split,
                "sample_id": sample.sample_id,
                "questions": sample.questions,
                "answers": sample.answers,
                "predictions": [],
                "metadata": sample.metadata,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.time() - started_at, 3),
            }
            if not continue_on_error:
                jsonl_append(output_path, record)
                raise
        finally:
            if temp_file is not None and temp_file.exists():
                temp_file.unlink()

        jsonl_append(output_path, record)
        completed_ids.add(sample_key)
        processed += 1

        if processed % log_every == 0:
            print(
                f"[{dataset_name}] processed={processed} skipped={skipped} errors={errors} "
                f"last_sample_id={sample.sample_id}",
                flush=True,
            )

    return {"processed": processed, "skipped": skipped, "errors": errors}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR-2 evaluation on ChartBench / ChartQA / ChartQAPro test splits."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASET_NAMES),
        help="Datasets to evaluate. Default: ChartBench ChartQA ChartQAPro.",
    )
    parser.add_argument("--split", default="test", help="Split to evaluate. Default: test.")
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help=f"Datasets root directory. Default: {DEFAULT_DATASETS_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jiaqiliu/scratch/532G/evaluation/results/deepseek_ocr2"),
        help="Directory to store per-dataset prediction JSONL files.",
    )
    parser.add_argument(
        "--temp-image-dir",
        type=Path,
        default=Path("/tmp/deepseek_ocr2_eval_images"),
        help="Directory used to materialize image bytes into temporary files.",
    )
    parser.add_argument(
        "--model-scratch-dir",
        type=Path,
        default=Path("/home/jiaqiliu/scratch/532G/evaluation/results/deepseek_ocr2/_model_scratch"),
        help="Scratch directory passed to model.infer(..., output_path=...).",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="HF model name.")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template. Must include '<image>' and '{question}'.",
    )
    parser.add_argument("--base-size", type=int, default=1024, help="DeepSeek-OCR-2 base_size.")
    parser.add_argument("--image-size", type=int, default=768, help="DeepSeek-OCR-2 image_size.")
    parser.add_argument(
        "--crop-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to enable DeepSeek-OCR-2 crop_mode. Default: true.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Value passed as _attn_implementation when loading the model.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype used after loading the model.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value to set before model loading.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="DuckDB parquet fetch batch size for ChartQA / ChartQAPro.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on processed samples per dataset for debugging.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print progress every N processed samples. Default: 10.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction files instead of resuming.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue after sample-level inference failures and record them to JSONL.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    if "<image>" not in args.prompt_template:
        parser.error("--prompt-template must include '<image>'.")
    if "{question}" not in args.prompt_template:
        parser.error("--prompt-template must include '{question}'.")

    runner = DeepSeekOCR2Runner(
        model_name=args.model_name,
        prompt_template=args.prompt_template,
        scratch_output_dir=args.model_scratch_dir,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
        attn_implementation=args.attn_implementation,
        dtype=args.dtype,
    )

    overall_exit_code = 0
    for dataset_name in args.datasets:
        output_path = build_output_path(args.output_dir, dataset_name, args.split)
        print(f"=== Evaluating {dataset_name} ({args.split}) ===", flush=True)
        print(f"output_file: {output_path}", flush=True)
        try:
            stats = evaluate_dataset(
                dataset_name=dataset_name,
                split=args.split,
                datasets_root=args.datasets_root,
                output_path=output_path,
                temp_image_dir=args.temp_image_dir,
                runner=runner,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                max_samples=args.max_samples,
                log_every=args.log_every,
                continue_on_error=args.continue_on_error,
            )
        except Exception as exc:
            overall_exit_code = 1
            print(f"FAILED on {dataset_name}: {exc}", file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            if not args.continue_on_error:
                return overall_exit_code
            continue

        print(
            f"finished {dataset_name}: processed={stats['processed']} skipped={stats['skipped']} "
            f"errors={stats['errors']}",
            flush=True,
        )

    return overall_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
