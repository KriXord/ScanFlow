"""Dynamic ScanFlow Pro V2 vLLM evaluation and representation analysis.

The normal pass produces the original per-dataset prediction JSONL files.  A
separate single-request analysis pass captures unambiguous V2 states and writes:

* per-question/per-view gate and realized-update statistics;
* benchmark CSV summaries and violin plots;
* bounded qualitative tensor payloads, spatial panels, and shared-PCA panels.

Preferred vllm_dir capture payload::

    {
        "v2_views": [
            {
                "view_type": "global",       # or "local"
                "grid_shape": [16, 16],
                "v2_original_visual": V,      # [N, 896]
                "v2_reasoning_residual": R,   # [N, 896]
                "v2_intensity": intensity,    # [N] or [N, 1]
                "v2_modified_visual": V_prime,# [N, 896]
                # Optional post-projector tensors:
                "v2_projected_original": Z,
                "v2_projected_modified": Z_prime,
            }
        ]
    }

For a single view, the same keys may be stored at payload root.  The runtime
must react to SCANFLOW_V2_SAVE_ANALYSIS/SCANFLOW_V2_ANALYSIS_DIR and preserve
the existing request-metadata queue contract.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from evaluation.dataloader import (
        DEFAULT_DATASETS_ROOT,
        ChartSample,
        iter_dataset,
        iter_salchartqa,
    )
    from evaluation.prompt_templates import (
        DEFAULT_CHARTBENCH_STATEMENT_PROMPT_TEMPLATE,
        DEFAULT_CHARTBENCH_VALUE_PROMPT_TEMPLATE,
        DEFAULT_CHARTQAPRO_CONVERSATIONAL_PROMPT_TEMPLATE,
        DEFAULT_CHARTQAPRO_FACT_CHECKING_PROMPT_TEMPLATE,
        DEFAULT_CHARTQAPRO_FACTOID_PROMPT_TEMPLATE,
        DEFAULT_CHARTQAPRO_HYPOTHETICAL_PROMPT_TEMPLATE,
        DEFAULT_CHARTQAPRO_MULTI_CHOICE_PROMPT_TEMPLATE,
        DEFAULT_PROMPT_TEMPLATE,
        build_chartqapro_conversation_history,
        is_chartbench_percentage_question,
        is_chartbench_value_question,
        normalize_chartqapro_question_type,
    )
    from evaluation.run_deepseek_ocr2_eval import (
        DEFAULT_DATASET_NAMES,
        build_output_path,
        jsonl_append,
        load_existing_sample_ids,
    )
except ImportError:
    from dataloader import (  # type: ignore
        DEFAULT_DATASETS_ROOT,
        ChartSample,
        iter_dataset,
        iter_salchartqa,
    )
    from prompt_templates import (  # type: ignore
        DEFAULT_CHARTBENCH_STATEMENT_PROMPT_TEMPLATE,
        DEFAULT_CHARTBENCH_VALUE_PROMPT_TEMPLATE,
        DEFAULT_CHARTQAPRO_CONVERSATIONAL_PROMPT_TEMPLATE,
        DEFAULT_CHARTQAPRO_FACT_CHECKING_PROMPT_TEMPLATE,
        DEFAULT_CHARTQAPRO_FACTOID_PROMPT_TEMPLATE,
        DEFAULT_CHARTQAPRO_HYPOTHETICAL_PROMPT_TEMPLATE,
        DEFAULT_CHARTQAPRO_MULTI_CHOICE_PROMPT_TEMPLATE,
        DEFAULT_PROMPT_TEMPLATE,
        build_chartqapro_conversation_history,
        is_chartbench_percentage_question,
        is_chartbench_value_question,
        normalize_chartqapro_question_type,
    )
    from run_deepseek_ocr2_eval import (  # type: ignore
        DEFAULT_DATASET_NAMES,
        build_output_path,
        jsonl_append,
        load_existing_sample_ids,
    )


DEFAULT_DEEPSEEK_VLLM_DIR = Path(
    "/home/krisxu/links/scratch/2026_summer_research_project/CueFlow/DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm"
)

DEFAULT_V2_MODEL_NAME = (
    "/home/krisxu/links/scratch/2026_summer_research_project/CueFlow/output/"
    "scanflow_pro_v2_dynamic_intensity_attention/"
    "v0-20260902-151808/checkpoint-3600"
)

DEFAULT_OUTPUT_DIR = Path(
    "/home/krisxu/links/scratch/2026_summer_research_project/CueFlow/eval_outputs/"
    "scanflow_pro_v2_dynamic_analysis/checkpoint-3600_vllm"
)

# Start conservatively for ScanFlow. The SLURM script below uses 1 for the
# first compatibility run; this default can be raised after that succeeds.
DEFAULT_MAX_NUM_SEQS = 4
DEFAULT_GPU_MEMORY_UTILIZATION = 0.75
DEFAULT_MAX_TOKENS = 128

SCANFLOW_MODEL_REGISTRY_NAME = "ScanflowProDeepseekOCR2ForCausalLM"

V2_WEIGHT_PREFIXES = (
    "scanflow_v2_residual_attn",
    "scanflow_v2_intensity_mlp",
)

V2_SOURCE_MARKERS = (
    "scanflow_v2_residual_attn",
    "scanflow_v2_intensity_mlp",
    "scanflow_v2_intensity_visual_norm",
    "scanflow_v2_intensity_question_norm",
)

# The vLLM runtime must save these values in each capture payload. Aliases let
# the evaluator tolerate minor naming differences while vllm_dir is updated.
V2_CAPTURE_ALIASES = {
    "original": ("v2_original_visual", "original_visual_tokens", "visual_tokens", "V"),
    "residual": ("v2_reasoning_residual", "reasoning_residual", "residual", "R"),
    "intensity": ("v2_intensity", "intensity", "lambda_map", "lambda"),
    "modified": ("v2_modified_visual", "modified_visual_tokens", "V_modified", "V_prime"),
    "projected_original": ("v2_projected_original", "projected_original_visual"),
    "projected_modified": ("v2_projected_modified", "projected_modified_visual"),
}


def prepend_env_path(var_name: str, path: Path) -> None:
    if not path.is_dir():
        return
    current = os.environ.get(var_name, "")
    values = [value for value in current.split(os.pathsep) if value]
    path_str = str(path)
    if path_str in values:
        return
    os.environ[var_name] = path_str if not values else f"{path_str}{os.pathsep}{current}"


def configure_cuda_runtime(cuda_version: str | None) -> None:
    if not cuda_version:
        return

    candidate_roots: list[Path] = []
    if cuda_version == "11.8":
        candidate_roots.extend(
            [
                Path("/cm/shared/apps/cuda11.8/toolkit/11.8.0"),
                Path("/usr/local/cuda-11.8"),
                Path("/usr/local/cuda"),
            ]
        )

    for root in candidate_roots:
        lib_candidates = [
            root / "targets/x86_64-linux/lib",
            root / "lib64",
            root / "lib",
        ]
        bin_candidates = [
            root / "bin",
        ]
        libcudart_found = any((lib_dir / "libcudart.so.11.0").exists() for lib_dir in lib_candidates)
        if not libcudart_found:
            continue

        os.environ.setdefault("CUDA_HOME", str(root))
        os.environ.setdefault("CUDA_PATH", str(root))
        os.environ.setdefault("CUDA_ROOT", str(root))
        for lib_dir in lib_candidates:
            prepend_env_path("LD_LIBRARY_PATH", lib_dir)
            prepend_env_path("LIBRARY_PATH", lib_dir)
        for bin_dir in bin_candidates:
            prepend_env_path("PATH", bin_dir)

        ptxas_candidates = [
            root / "bin/ptxas",
            root / "targets/x86_64-linux/bin/ptxas",
        ]
        for ptxas_path in ptxas_candidates:
            if ptxas_path.exists():
                os.environ.setdefault("TRITON_PTXAS_PATH", str(ptxas_path))
                break
        return


def validate_transformers_stack() -> None:
    try:
        import tokenizers
        import transformers
    except ImportError:
        return

    transformers_version = getattr(transformers, "__version__", "")
    tokenizers_version = getattr(tokenizers, "__version__", "")

    try:
        transformers_major = int(transformers_version.split(".", 1)[0])
    except Exception:
        transformers_major = None

    try:
        tokenizers_major, tokenizers_minor = (
            int(part) for part in tokenizers_version.split(".")[:2]
        )
    except Exception:
        tokenizers_major = None
        tokenizers_minor = None

    if transformers_major is not None and transformers_major >= 5:
        raise RuntimeError(
            "This DeepSeek-OCR2 vLLM setup is not compatible with transformers "
            f"{transformers_version}. Please install `transformers==4.46.3` and "
            "`tokenizers==0.20.3` in the evaluation environment."
        )

    if (
        tokenizers_major is not None
        and tokenizers_minor is not None
        and (tokenizers_major, tokenizers_minor) >= (0, 23)
    ):
        raise RuntimeError(
            "This DeepSeek-OCR2 vLLM setup is not compatible with tokenizers "
            f"{tokenizers_version}. Please install `transformers==4.46.3` and "
            "`tokenizers==0.20.3` in the evaluation environment."
        )



def resolve_latest_checkpoint_dir(model_name: str) -> str:
    """Resolve an ms-swift run directory to its latest checkpoint-* child.

    If ``model_name`` already points at a self-contained model/checkpoint, it is
    returned unchanged. This lets the requested v1-... run directory be used
    directly while still loading its final checkpoint for vLLM.
    """
    candidate = Path(model_name).expanduser()
    if not candidate.is_dir():
        return model_name

    model_markers = (
        "model.safetensors.index.json",
        "model.safetensors",
        "pytorch_model.bin",
        "config.json",
    )
    has_weights = any((candidate / marker).is_file() for marker in model_markers[:3])
    if has_weights:
        return str(candidate.resolve())

    checkpoint_dirs: list[tuple[int, Path]] = []
    for child in candidate.glob("checkpoint-*"):
        if not child.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", child.name)
        if match is None:
            continue
        checkpoint_dirs.append((int(match.group(1)), child))

    if not checkpoint_dirs:
        # Return the original directory; downstream loading will emit the useful
        # error if it is not actually a model directory.
        return str(candidate.resolve())

    checkpoint_dirs.sort(key=lambda item: item[0])
    resolved = checkpoint_dirs[-1][1].resolve()
    print(
        f"Resolved ms-swift run directory {candidate} -> {resolved}",
        flush=True,
    )
    return str(resolved)


def checkpoint_is_plan1_parallel(model_name: str) -> bool:
    """Detect the fixed-query Plan 1 Parallel architecture from checkpoint files."""
    model_dir = resolve_local_model_dir(model_name)
    if model_dir is None:
        return False

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            with index_path.open("r", encoding="utf-8") as handle:
                weight_map = json.load(handle).get("weight_map", {})
        except Exception:
            weight_map = {}

        if weight_map:
            return all(
                any(name.startswith(prefix) for name in weight_map)
                for prefix in PLAN1_PARALLEL_WEIGHT_PREFIXES
            )

    # Fallback for checkpoints without an index file.
    for source_name in ("deepencoderv2.py", "modeling_deepseekocr2.py"):
        source_path = model_dir / source_name
        if not source_path.is_file():
            continue
        try:
            source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if all(marker in source_text for marker in ("query_768", "query_1024")):
            return True

    return False


def resolve_local_model_dir(model_name: str) -> Path | None:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return candidate.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None

    try:
        return Path(snapshot_download(repo_id=model_name, local_files_only=True)).resolve()
    except Exception:
        return None


def model_uses_text_query_residual(model_name: str) -> bool:
    return detect_text_query_residual_mode(model_name) != "none"


def detect_text_query_residual_mode(model_name: str) -> str:
    """Return one of: scanflow, full, residual_only, none."""
    model_dir = resolve_local_model_dir(model_name)
    if model_dir is None:
        return "none"

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as handle:
                weight_map = json.load(handle).get("weight_map", {})
        except Exception:
            weight_map = {}

        has_scanflow = any(
            tensor_name.startswith(SCANFLOW_WEIGHT_PREFIXES)
            for tensor_name in weight_map
        )
        has_text_context = any(
            tensor_name.startswith(TEXT_QUERY_RESIDUAL_ONLY_PREFIXES)
            for tensor_name in weight_map
        )
        has_full = any(
            tensor_name.startswith(TEXT_QUERY_RESIDUAL_WEIGHT_PREFIXES[1:])
            for tensor_name in weight_map
        )

        if has_scanflow:
            return "scanflow"
        if has_full:
            return "full"
        if has_text_context:
            return "residual_only"

    args_path = model_dir / "args.json"
    if args_path.exists():
        try:
            args_text = args_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            args_text = ""

        if any(marker in args_text for marker in SCANFLOW_SOURCE_MARKERS):
            return "scanflow"

        if "text_context_proj" in args_text:
            if (
                "text_query_mlp" in args_text
                or "text_query_residual_scale" in args_text
            ):
                return "full"
            return "residual_only"

    source_files = (
        model_dir / "deepencoderv2.py",
        model_dir / "modeling_deepseekocr2.py",
    )
    source_texts: list[str] = []

    for source_path in source_files:
        if not source_path.exists():
            continue
        try:
            source_texts.append(
                source_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            )
        except Exception:
            continue

    for source_text in source_texts:
        if any(marker in source_text for marker in SCANFLOW_SOURCE_MARKERS):
            return "scanflow"

        if "text_context_proj" in source_text:
            if (
                "text_query_mlp" in source_text
                or "text_query_residual_scale" in source_text
            ):
                return "full"
            if (
                "_apply_text_context" in source_text
                or "enable_text_query_residual" in source_text
            ):
                return "residual_only"

    return "none"



def configure_plan1_parallel_inference_environment() -> None:
    """Reconstruct the fixed-query Plan 1 Parallel architecture for QA inference.

    This mirrors the architecture-faithful Transformers evaluator:
      * fixed 144/256 causal query layout
      * no Plan-2 recurrent controller
      * no dynamic causal-token template
      * no auxiliary scanpath/attention losses during ChartQA generation
      * no parameter reset
    """
    values = {
        "SCANFLOW_PLAN2": "0",
        "SCANFLOW_DYNAMIC_CAUSAL_TOKENS": "0",
        "SCANFLOW_RESET_NEW_PARAMS": "0",

        # Plan 1 Parallel does NOT use the old text-query residual MLP switch.
        "DEEPSEEK_OCR2_ENABLE_TEXT_QUERY_RESIDUAL": "0",
        "DEEPSEEK_OCR2_TEXT_QUERY_RESIDUAL_ONLY": "0",

        # QA-only vLLM generation: architecture stays intact; supervision is off.
        "DEEPSEEK_OCR2_ENABLE_ATTENTION_SUPERVISION": "0",
        "DEEPSEEK_OCR2_FIXATION_LOSS_WEIGHT": "0.0",
        "DEEPSEEK_OCR2_DIVERSITY_LOSS_WEIGHT": "0.0",
        "DEEPSEEK_OCR2_ATTENTION_LOSS_WEIGHT": "0.0",
        "DEEPSEEK_OCR2_ATTENTION_DIVERSITY_LOSS_WEIGHT": "0.0",
        "SCANFLOW_PLAN1_PREDICTOR_LOSS_WEIGHT": "0.0",

        "DEEPSEEK_OCR2_COMPUTE_FIXATION_DIAGNOSTICS": "0",
        "DEEPSEEK_OCR2_COMPUTE_ATTENTION_DIAGNOSTICS": "0",
        "DEEPSEEK_OCR2_COMPUTE_ATTENTION_DIVERSITY_DIAGNOSTICS": "0",

        "CROP_MODE": "1",
        "BASE_SIZE": "1024",
        "IMAGE_SIZE": "768",
    }
    for key, value in values.items():
        os.environ[key] = value


def validate_plan1_parallel_vllm_runtime(deepseek_vllm_dir: Path) -> None:
    """Fail early if the external vLLM runtime still implements an older ScanFlow.

    vLLM executes scanflow_pro_deepseek_ocr2.py / scanflow_pro_deepencoder.py rather than
    importing the checkpoint's Hugging Face deepencoderv2.py directly. Therefore
    the external runtime must itself contain the Plan 1 Parallel fixed-query path.
    """
    wrapper_path = deepseek_vllm_dir / "scanflow_pro_deepseek_ocr2.py"
    encoder_path = deepseek_vllm_dir / "scanflow_pro_deepencoder.py"

    missing_files = [
        str(path)
        for path in (wrapper_path, encoder_path)
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Required Plan 1 Parallel vLLM runtime file(s) are missing: "
            + ", ".join(missing_files)
        )

    wrapper_text = wrapper_path.read_text(encoding="utf-8", errors="ignore")
    encoder_text = encoder_path.read_text(encoding="utf-8", errors="ignore")
    combined = wrapper_text + "\n" + encoder_text

    required_markers = ("query_768", "query_1024")
    missing_markers = [
        marker for marker in required_markers if marker not in combined
    ]
    if missing_markers:
        raise RuntimeError(
            "The external DeepSeek-OCR2 vLLM runtime is NOT Plan 1 Parallel compatible. "
            "It is missing fixed-query marker(s): "
            + ", ".join(missing_markers)
            + ". The evaluator intentionally refuses to run this checkpoint through "
              "the older recurrent ScanFlow runtime because that would change the "
              "model architecture. Update scanflow_pro_deepencoder.py / "
              "scanflow_pro_deepseek_ocr2.py to the requested implementation first."
        )

    # Predictor support is not needed for QA-only generation, but report whether
    # the external runtime has it. Behavioral predictor visualization remains a
    # Transformers-evaluator responsibility unless explicitly ported to vLLM.
    predictor_available = "scanpath_predictor" in combined
    print(
        "Plan 1 Parallel vLLM runtime preflight passed: "
        "fixed query_768/query_1024 path found; "
        f"scanpath_predictor_runtime={int(predictor_available)}.",
        flush=True,
    )


def configure_scanflow_inference_environment() -> None:
    """
    Reconstruct the latest attention-alignment/diversity ScanFlow architecture
    for QA-only vLLM inference.

    Auxiliary loss weights stay at zero because ChartQA/ChartQAPro generation
    has no fixation targets. Attention-map collection is disabled by default
    here because this pipeline is for fast answer generation/accuracy; turning
    it on does not change the learned parameters.
    """
    defaults = {
        "DEEPSEEK_OCR2_ENABLE_TEXT_QUERY_RESIDUAL": "1",
        "DEEPSEEK_OCR2_TEXT_QUERY_RESIDUAL_ONLY": "0",
        "DEEPSEEK_OCR2_SCANPATH_START_SOURCE": "visual_mean",
        "SCANFLOW_RESET_NEW_PARAMS": "0",

        # QA-only inference: preserve architecture, disable auxiliary losses.
        "DEEPSEEK_OCR2_FIXATION_LOSS_WEIGHT": "0.0",
        "DEEPSEEK_OCR2_DIVERSITY_LOSS_WEIGHT": "0.0",
        "DEEPSEEK_OCR2_ATTENTION_LOSS_WEIGHT": "0.0",
        "DEEPSEEK_OCR2_ATTENTION_DIVERSITY_LOSS_WEIGHT": "0.0",

        # We only need predictions in this pipeline, not attention diagnostics.
        "DEEPSEEK_OCR2_ENABLE_ATTENTION_SUPERVISION": "0",
        "DEEPSEEK_OCR2_COMPUTE_FIXATION_DIAGNOSTICS": "0",
        "DEEPSEEK_OCR2_COMPUTE_ATTENTION_DIAGNOSTICS": "0",
        "DEEPSEEK_OCR2_COMPUTE_ATTENTION_DIVERSITY_DIAGNOSTICS": "0",

        "DEEPSEEK_OCR2_ATTENTION_LAYER_INDEX": "-1",
        "DEEPSEEK_OCR2_ATTENTION_FIX_HEADS": "2",

        "CROP_MODE": "1",
        "BASE_SIZE": "1024",
        "IMAGE_SIZE": "768",
    }

    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def validate_scanflow_runtime_files(
    deepseek_vllm_dir: Path,
    *,
    require_latest_scanflow: bool,
) -> None:
    """
    Fail early when the out-of-tree vLLM runtime is older than the checkpoint.

    vLLM executes scanflow_pro_deepseek_ocr2.py / scanflow_pro_deepencoder.py after
    ModelRegistry registration; it does not automatically execute the
    checkpoint's Hugging Face deepencoderv2.py implementation.
    """
    wrapper_path = deepseek_vllm_dir / "scanflow_pro_deepseek_ocr2.py"
    encoder_path = deepseek_vllm_dir / "scanflow_pro_deepencoder.py"

    missing = [
        str(path)
        for path in (wrapper_path, encoder_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required ScanFlow vLLM runtime file(s) are missing: "
            + ", ".join(missing)
        )

    wrapper_text = wrapper_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )
    encoder_text = encoder_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if "scanflow_pro_deepencoder" not in wrapper_text:
        raise RuntimeError(
            "scanflow_pro_deepseek_ocr2.py is not wired to "
            "scanflow_pro_deepencoder.py."
        )

    if not require_latest_scanflow:
        return

    required_encoder_markers = (
        "scanpath_blocks",
        "scanpath_input_norm",
        "scanpath_to_query_projector",
        "text_query_out",
        "fixation_decoder",
        "scanpath_start_source",
        "visual_mean",
    )
    missing_markers = [
        marker
        for marker in required_encoder_markers
        if marker not in encoder_text
    ]

    if missing_markers:
        raise RuntimeError(
            "The vLLM scanflow_pro_deepencoder.py does not match the latest "
            "visual-mean ScanFlow architecture. Missing marker(s): "
            + ", ".join(missing_markers)
            + ". Update the vLLM ScanFlow runtime before evaluating this "
              "checkpoint."
        )

    # The attention-diversity checkpoint was trained with an attention-capable
    # recurrent block. The diversity loss itself introduces no new inference
    # parameter, but the runtime should support the attention-supervision path.
    if (
        "enable_attention_supervision" not in encoder_text
        and "DEEPSEEK_OCR2_ENABLE_ATTENTION_SUPERVISION" not in encoder_text
    ):
        raise RuntimeError(
            "The vLLM scanflow_pro_deepencoder.py lacks the attention-capable "
            "ScanFlow path used by this checkpoint."
        )

    print(
        "ScanFlow vLLM runtime preflight passed: "
        "visual-mean start + attention-capable recurrent architecture found.",
        flush=True,
    )


def patch_qwen2_residual_only_mode() -> None:
    try:
        from deepencoderv2 import qwen2_d2e
    except Exception:
        return

    if getattr(qwen2_d2e, "_codex_residual_only_patch", False):
        return

    original_init = qwen2_d2e.Qwen2Decoder2Encoder.__init__
    original_apply = qwen2_d2e.Qwen2Decoder2Encoder._apply_text_context

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if os.getenv("DEEPSEEK_OCR2_TEXT_QUERY_RESIDUAL_ONLY", "0") != "1":
            return
        # Remove the MLP + scale parameters so weight loading doesn't require them.
        if getattr(self, "text_query_mlp", None) is not None:
            self.text_query_mlp = None
        if getattr(self, "text_query_residual_scale", None) is not None:
            self.text_query_residual_scale = None
        # Ensure the flag stays on so text_context_proj path is used.
        self.enable_text_query_residual = True

    def patched_apply(self, base_query_imgs, text_context):
        if (
            not getattr(self, "enable_text_query_residual", False)
            or text_context is None
            or getattr(self, "text_context_proj", None) is None
        ):
            return base_query_imgs

        if (
            getattr(self, "text_query_mlp", None) is None
            or getattr(self, "text_query_residual_scale", None) is None
        ):
            residual = self.text_context_proj(text_context).to(base_query_imgs.dtype).unsqueeze(1)
            return base_query_imgs + residual

        return original_apply(self, base_query_imgs, text_context)

    qwen2_d2e.Qwen2Decoder2Encoder.__init__ = patched_init
    qwen2_d2e.Qwen2Decoder2Encoder._apply_text_context = patched_apply
    qwen2_d2e._codex_residual_only_patch = True


def checkpoint_is_v2_dynamic(model_name: str) -> bool:
    model_dir = resolve_local_model_dir(model_name)
    if model_dir is None:
        return False
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        except Exception:
            weight_map = {}
        if weight_map and all(
            any(prefix in name for name in weight_map)
            for prefix in V2_WEIGHT_PREFIXES
        ):
            return True
    for source_name in ("deepencoderv2.py", "modeling_deepseekocr2.py"):
        source_path = model_dir / source_name
        if source_path.is_file():
            source = source_path.read_text(encoding="utf-8", errors="ignore")
            if all(marker in source for marker in V2_SOURCE_MARKERS):
                return True
    return False


def configure_v2_inference_environment() -> None:
    values = {
        "SCANFLOW_PLAN2": "0",
        "SCANFLOW_DYNAMIC_CAUSAL_TOKENS": "0",
        "SCANFLOW_RESET_NEW_PARAMS": "0",
        "SCANFLOW_V2_RESET_NEW_PARAMS": "0",
        # Legacy name used only to reserve the shared N+T placeholder layout.
        "SCANFLOW_PRO_V4": "1",
        "DEEPSEEK_OCR2_ENABLE_TEXT_QUERY_RESIDUAL": "1",
        "DEEPSEEK_OCR2_TEXT_QUERY_RESIDUAL_ONLY": "0",
        "DEEPSEEK_OCR2_SCANPATH_START_SOURCE": "visual_mean",
        "DEEPSEEK_OCR2_ENABLE_ATTENTION_SUPERVISION": "0",
        "DEEPSEEK_OCR2_FIXATION_LOSS_WEIGHT": "0.0",
        "DEEPSEEK_OCR2_DIVERSITY_LOSS_WEIGHT": "0.0",
        "DEEPSEEK_OCR2_ATTENTION_LOSS_WEIGHT": "0.0",
        "DEEPSEEK_OCR2_ATTENTION_DIVERSITY_LOSS_WEIGHT": "0.0",
        "SCANFLOW_PLAN1_PREDICTOR_LOSS_WEIGHT": "0.0",
        "DEEPSEEK_OCR2_COMPUTE_FIXATION_DIAGNOSTICS": "0",
        "DEEPSEEK_OCR2_COMPUTE_ATTENTION_DIAGNOSTICS": "0",
        "DEEPSEEK_OCR2_COMPUTE_ATTENTION_DIVERSITY_DIAGNOSTICS": "0",
        "CROP_MODE": "1",
        "BASE_SIZE": "1024",
        "IMAGE_SIZE": "768",
    }
    for key, value in values.items():
        os.environ[key] = value


def validate_v2_vllm_runtime(deepseek_vllm_dir: Path, *, require_capture: bool) -> None:
    wrapper_path = deepseek_vllm_dir / "scanflow_pro_deepseek_ocr2.py"
    encoder_path = deepseek_vllm_dir / "scanflow_pro_deepencoder.py"
    missing_files = [str(path) for path in (wrapper_path, encoder_path) if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Required V2 vLLM runtime file(s) missing: " + ", ".join(missing_files))
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (wrapper_path, encoder_path)
    )
    missing_arch = [marker for marker in V2_SOURCE_MARKERS if marker not in combined]
    if missing_arch:
        raise RuntimeError(
            "vllm_dir is not dynamic ScanFlow Pro V2 compatible; missing: "
            + ", ".join(missing_arch)
        )
    capture_markers = ("SCANFLOW_V2_SAVE_ANALYSIS", "push_scanflow_request_metadata")
    missing_capture = [marker for marker in capture_markers if marker not in combined]
    if require_capture and missing_capture:
        raise RuntimeError(
            "V2 analysis capture is unavailable; missing runtime marker(s): "
            + ", ".join(missing_capture)
            + ". The runtime must save V, R, lambda, and V_modified per prefill."
        )
    print(
        "ScanFlow Pro V2 runtime preflight passed: "
        f"dynamic_intensity=1 capture_ready={int(not missing_capture)}.",
        flush=True,
    )


@dataclass
class PendingQuestion:
    sample_key: str
    question_idx: int
    request: dict[str, Any]


@dataclass
class SampleState:
    dataset: str
    split: str
    sample_id: int | str
    questions: list[str]
    answers: list[str]
    predictions: list[str | None]
    question_errors: list[str | None]
    metadata: dict[str, Any]
    image: Any
    started_at: float
    remaining: int
    next_question_idx_to_schedule: int = 0


@dataclass
class ScanFlowVisualizationTask:
    dataset: str
    split: str
    sample_id: int | str
    question_idx: int
    question: str
    answer: str
    prediction: str
    metadata: dict[str, Any]
    request: dict[str, Any]
    image: Any


class ScanFlowVisualizationCollector:
    """
    Collect only a bounded number of question-level requests during the normal
    vLLM evaluation.

    No extra model work happens here. The selected requests are re-run only
    after the normal prediction workflow has completely finished.
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(0, int(limit))
        self.tasks: list[ScanFlowVisualizationTask] = []

    def maybe_add(
        self,
        *,
        task: PendingQuestion,
        state: SampleState,
    ) -> None:
        if self.limit <= 0 or len(self.tasks) >= self.limit:
            return
        if state.image is None:
            return

        prediction = state.predictions[task.question_idx]
        if prediction is None:
            return

        answer = (
            state.answers[task.question_idx]
            if task.question_idx < len(state.answers)
            else ""
        )

        try:
            image_copy = state.image.copy()
        except Exception:
            return

        self.tasks.append(
            ScanFlowVisualizationTask(
                dataset=state.dataset,
                split=state.split,
                sample_id=state.sample_id,
                question_idx=task.question_idx,
                question=state.questions[task.question_idx],
                answer=str(answer),
                prediction=str(prediction),
                metadata=dict(state.metadata),
                request=task.request,
                image=image_copy,
            )
        )


def iter_v2_evaluation_samples(
    *,
    dataset_name: str,
    split: str,
    datasets_root: Path,
    batch_size: int,
    salchartqa_annotation_file: str | None,
):
    """Use SalChartQA's explicit JSONL loader and the shared loader otherwise."""
    dataset_key = dataset_name.strip().lower().replace("-", "").replace("_", "")
    if dataset_key in {"salchartqa", "salchartqascanpath"}:
        return iter_salchartqa(
            split=split,
            dataset_root=datasets_root / "SalChartQA",
            include_image_bytes=True,
            include_raw=True,
            annotation_file=salchartqa_annotation_file,
        )
    return iter_dataset(
        dataset_name=dataset_name,
        split=split,
        datasets_root=datasets_root,
        include_image_bytes=True,
        batch_size=batch_size,
    )


def load_sample_pil_image(sample: ChartSample):
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("vLLM evaluation requires Pillow (PIL).") from exc

    if sample.image_path:
        image_path = Path(sample.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        with Image.open(image_path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")

    if sample.image_bytes is None:
        raise ValueError(
            f"Sample {sample.sample_id} from {sample.dataset} has neither image_path nor image_bytes."
        )

    with Image.open(io.BytesIO(sample.image_bytes)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def clean_generated_text(text: str) -> str:
    stop_str = "<｜end▁of▁sentence｜>"
    if text.endswith(stop_str):
        text = text[: -len(stop_str)]
    return text.strip()


def extract_first_number(text: str, *, keep_percent: bool = False) -> str | None:
    pattern = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?" if keep_percent else r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    match = re.search(pattern, text)
    if match is None:
        return None
    return match.group(0).replace(",", "")


def normalize_prediction(
    dataset_name: str,
    question: str,
    text: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    cleaned = clean_generated_text(text)
    if dataset_name == "ChartQAPro":
        question_type = normalize_chartqapro_question_type((metadata or {}).get("question_type"))
        if question_type == "multi choice":
            match = re.search(r"\b([abcd])\b", cleaned, flags=re.IGNORECASE)
            return match.group(1).lower() if match is not None else cleaned
        if question_type == "fact checking":
            match = re.search(r"\b(true|false)\b", cleaned, flags=re.IGNORECASE)
            return match.group(1).lower() if match is not None else cleaned
        return cleaned

    if dataset_name != "ChartBench":
        return cleaned

    if is_chartbench_value_question(question):
        first_number = extract_first_number(
            cleaned,
            keep_percent=is_chartbench_percentage_question(question),
        )
        return first_number if first_number is not None else cleaned

    match = re.search(r"\b(yes|no|true|false)\b", cleaned, flags=re.IGNORECASE)
    if match is None:
        return cleaned

    return "Yes" if match.group(1).lower() in {"yes", "true"} else "No"


def is_chartqapro_conversational_sample(dataset_name: str, metadata: dict[str, Any]) -> bool:
    return (
        dataset_name == "ChartQAPro"
        and normalize_chartqapro_question_type(metadata.get("question_type")) == "conversational"
    )


class DeepSeekOCR2VLLMRunner:
    def __init__(
        self,
        *,
        deepseek_vllm_dir: Path,
        model_name: str,
        prompt_template: str,
        chartbench_statement_prompt_template: str,
        chartbench_value_prompt_template: str,
        chartqapro_factoid_prompt_template: str,
        chartqapro_multi_choice_prompt_template: str,
        chartqapro_hypothetical_prompt_template: str,
        chartqapro_fact_checking_prompt_template: str,
        chartqapro_conversational_prompt_template: str,
        dtype: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_num_seqs: int,
        block_size: int,
        enforce_eager: bool,
        text_query_residual: bool | None,
        disable_mm_preprocessor_cache: bool,
        max_tokens: int,
        temperature: float,
        use_no_repeat_ngram: bool,
        no_repeat_ngram_size: int,
        no_repeat_window_size: int,
        architecture: str = "scanflow-pro-v2",
        collect_visual_attention: bool = False,
    ) -> None:
        self.deepseek_vllm_dir = deepseek_vllm_dir
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.chartbench_statement_prompt_template = chartbench_statement_prompt_template
        self.chartbench_value_prompt_template = chartbench_value_prompt_template
        self.chartqapro_factoid_prompt_template = chartqapro_factoid_prompt_template
        self.chartqapro_multi_choice_prompt_template = chartqapro_multi_choice_prompt_template
        self.chartqapro_hypothetical_prompt_template = chartqapro_hypothetical_prompt_template
        self.chartqapro_fact_checking_prompt_template = chartqapro_fact_checking_prompt_template
        self.chartqapro_conversational_prompt_template = chartqapro_conversational_prompt_template
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.block_size = block_size
        self.enforce_eager = enforce_eager
        self.text_query_residual = text_query_residual
        self.disable_mm_preprocessor_cache = disable_mm_preprocessor_cache
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_no_repeat_ngram = use_no_repeat_ngram
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.no_repeat_window_size = no_repeat_window_size
        self.architecture = architecture
        self.collect_visual_attention = bool(collect_visual_attention)
        self.llm = None
        self.sampling_params = None
        self.image_processor = None
        self.crop_mode = True

    def shutdown(self) -> None:
        llm = self.llm
        self.llm = None
        self.sampling_params = None
        self.image_processor = None

        if llm is not None:
            try:
                llm_engine = getattr(llm, "llm_engine", None)
                model_executor = getattr(llm_engine, "model_executor", None)
                if model_executor is not None:
                    model_executor.shutdown()
            except Exception:
                pass

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass

    def load_image_processor(self) -> None:
        if self.image_processor is not None:
            return

        validate_transformers_stack()

        if str(self.deepseek_vllm_dir) not in sys.path:
            sys.path.insert(0, str(self.deepseek_vllm_dir))

        try:
            from config import CROP_MODE
            from process.image_process import DeepseekOCR2Processor
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import DeepSeek-OCR2 image processor from the vLLM repository. "
                f"Original import error: {exc}"
            ) from exc

        self.crop_mode = CROP_MODE
        self.image_processor = DeepseekOCR2Processor()

    def load(self) -> None:
        if self.llm is not None and self.sampling_params is not None:
            return

        if not self.deepseek_vllm_dir.exists():
            raise FileNotFoundError(f"DeepSeek-OCR2-vllm directory not found: {self.deepseek_vllm_dir}")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("vLLM evaluation requires torch.") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("vLLM evaluation requires CUDA, but no GPU is visible.")

        configure_cuda_runtime(torch.version.cuda)
        validate_transformers_stack()

        os.environ.setdefault("VLLM_USE_V1", "0")

        if str(self.deepseek_vllm_dir) not in sys.path:
            sys.path.insert(0, str(self.deepseek_vllm_dir))

        if not checkpoint_is_v2_dynamic(self.model_name):
            raise RuntimeError(
                "The checkpoint was not confirmed as dynamic ScanFlow Pro V2: "
                f"{self.model_name}"
            )
        validate_v2_vllm_runtime(
            self.deepseek_vllm_dir,
            require_capture=self.collect_visual_attention,
        )
        configure_v2_inference_environment()
        residual_mode = "scanflow-pro-v2-dynamic"
        enable_text_query_residual = True
        effective_enforce_eager = True

        print(
            "scanflow_runtime: "
            f"architecture={self.architecture} "
            f"residual_mode={residual_mode} "
            f"enable_text_query_residual={int(enable_text_query_residual)} "
            f"plan2={os.environ.get('SCANFLOW_PLAN2', 'unset')} "
            f"dynamic_causal_tokens={os.environ.get('SCANFLOW_DYNAMIC_CAUSAL_TOKENS', 'unset')} "
            f"attention_supervision={os.environ.get('DEEPSEEK_OCR2_ENABLE_ATTENTION_SUPERVISION', 'unset')} "
            f"enforce_eager={int(effective_enforce_eager)}",
            flush=True,
        )

        try:
            # from deepseek_ocr2 import DeepseekOCR2ForCausalLM
            from scanflow_pro_deepseek_ocr2 import DeepseekOCR2ForCausalLM
            from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
            from vllm import LLM, SamplingParams
            from vllm.model_executor.models.registry import ModelRegistry
        except ImportError as exc:
            if "libcudart.so.11.0" in str(exc):
                raise RuntimeError(
                    "vLLM import failed because CUDA 11.8 runtime libraries are not on "
                    "LD_LIBRARY_PATH. On this cluster, load `cuda11.8/toolkit/11.8.0` "
                    "or prepend `/cm/shared/apps/cuda11.8/toolkit/11.8.0/targets/x86_64-linux/lib` "
                    "before running the script."
                ) from exc
            raise RuntimeError(
                "ScanFlow vLLM evaluation requires vllm, transformers, "
                "scanflow_pro_deepseek_ocr2.py, scanflow_pro_deepencoder.py, and the "
                "custom DeepSeek-OCR2 modules. "
                f"Original import error: {exc}"
            ) from exc

        try:
            ModelRegistry.register_model(
                SCANFLOW_MODEL_REGISTRY_NAME,
                DeepseekOCR2ForCausalLM,
            )
        except Exception as exc:
            print(
                f"Warning: failed to register "
                f"{SCANFLOW_MODEL_REGISTRY_NAME}: {exc}",
                flush=True,
            )

        print(
            "===========================ScanFlow model registered: "
            f"{self.model_name}===========================",
            flush=True,
        )

        self.llm = LLM(
            model=self.model_name,
            hf_overrides={
                "architectures": [SCANFLOW_MODEL_REGISTRY_NAME]
            },
            block_size=self.block_size,
            enforce_eager=effective_enforce_eager,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            swap_space=0,
            max_num_seqs=self.max_num_seqs,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            dtype=self.dtype,
        )

        logits_processors = None
        if self.use_no_repeat_ngram:
            logits_processors = [
                NoRepeatNGramLogitsProcessor(
                    ngram_size=self.no_repeat_ngram_size,
                    window_size=self.no_repeat_window_size,
                    whitelist_token_ids={128821, 128822},
                )
            ]

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

    def get_prompt_template(
        self,
        dataset_name: str,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if dataset_name == "ChartBench":
            if is_chartbench_value_question(question):
                return self.chartbench_value_prompt_template
            return self.chartbench_statement_prompt_template

        if dataset_name == "ChartQAPro":
            question_type = normalize_chartqapro_question_type((metadata or {}).get("question_type"))
            if question_type == "factoid":
                return self.chartqapro_factoid_prompt_template
            if question_type == "multi choice":
                return self.chartqapro_multi_choice_prompt_template
            if question_type == "hypothetical":
                return self.chartqapro_hypothetical_prompt_template
            if question_type == "fact checking":
                return self.chartqapro_fact_checking_prompt_template
            if question_type == "conversational":
                return self.chartqapro_conversational_prompt_template
        return self.prompt_template

    def build_request(
        self,
        *,
        dataset_name: str,
        image: Any,
        question: str,
        metadata: dict[str, Any] | None = None,
        conversation: str | None = None,
    ) -> dict[str, Any]:
        self.load_image_processor()

        prompt_template = self.get_prompt_template(dataset_name, question, metadata=metadata)
        prompt = prompt_template.format(
            question=question.strip(),
            conversation=(conversation or "").strip(),
        )
        if "<image>" not in prompt:
            raise ValueError("prompt_template must contain the '<image>' placeholder.")

        image_features = self.image_processor.tokenize_with_images(  # type: ignore[union-attr]
            images=[image],
            conversation=prompt,
            bos=True,
            eos=True,
            cropping=self.crop_mode,
        )

        return {
            "prompt": prompt,
            "multi_modal_data": {"image": image_features},
        }

    def generate(self, requests: list[dict[str, Any]]) -> list[str]:
        self.load()
        outputs = self.llm.generate(
            requests,
            sampling_params=self.sampling_params,
        )  # type: ignore[union-attr]
        return [
            clean_generated_text(output.outputs[0].text)
            for output in outputs
        ]

    def generate_with_scanflow_capture(
        self,
        *,
        request: dict[str, Any],
        metadata: dict[str, Any],
        tensor_dir: Path,
    ) -> tuple[str, Path | None]:
        """
        Re-run one already-selected request and capture the ScanFlow tensors
        produced by the vLLM multimodal prefill.

        This method is used only by the post-evaluation visualization pass.
        Normal evaluation continues to call generate() exactly as before.
        """
        self.load()
        tensor_dir.mkdir(parents=True, exist_ok=True)

        try:
            import scanflow_pro_deepseek_ocr2 as scanflow_runtime
        except ImportError as exc:
            raise RuntimeError(
                "Could not import scanflow_pro_deepseek_ocr2 for visualization "
                "tensor capture."
            ) from exc

        before = set(tensor_dir.glob("*.pt"))

        old_save = os.environ.get("SCANFLOW_V2_SAVE_ANALYSIS")
        old_dir = os.environ.get("SCANFLOW_V2_ANALYSIS_DIR")
        old_metadata = os.environ.get("SCANFLOW_SAVE_ONLY_WITH_METADATA")

        try:
            os.environ["SCANFLOW_V2_SAVE_ANALYSIS"] = "1"
            os.environ["SCANFLOW_V2_ANALYSIS_DIR"] = str(tensor_dir)
            os.environ["SCANFLOW_SAVE_ONLY_WITH_METADATA"] = "1"
            os.environ["SCANFLOW_V2_INFERENCE_STEPS"] = "32"

            if hasattr(scanflow_runtime, "clear_scanflow_request_metadata_queue"):
                scanflow_runtime.clear_scanflow_request_metadata_queue()
            if hasattr(scanflow_runtime, "push_scanflow_request_metadata"):
                scanflow_runtime.push_scanflow_request_metadata(metadata)

            outputs = self.llm.generate(  # type: ignore[union-attr]
                [request],
                sampling_params=self.sampling_params,
            )
            prediction = clean_generated_text(outputs[0].outputs[0].text)
        finally:
            if hasattr(scanflow_runtime, "clear_scanflow_request_metadata_queue"):
                scanflow_runtime.clear_scanflow_request_metadata_queue()

            if old_save is None:
                os.environ.pop("SCANFLOW_V2_SAVE_ANALYSIS", None)
            else:
                os.environ["SCANFLOW_V2_SAVE_ANALYSIS"] = old_save

            if old_dir is None:
                os.environ.pop("SCANFLOW_V2_ANALYSIS_DIR", None)
            else:
                os.environ["SCANFLOW_V2_ANALYSIS_DIR"] = old_dir

            if old_metadata is None:
                os.environ.pop("SCANFLOW_SAVE_ONLY_WITH_METADATA", None)
            else:
                os.environ["SCANFLOW_SAVE_ONLY_WITH_METADATA"] = old_metadata

        after = set(tensor_dir.glob("*.pt"))
        new_files = sorted(
            after - before,
            key=lambda path: path.stat().st_mtime_ns,
        )
        tensor_path = new_files[-1] if new_files else None
        return prediction, tensor_path



def safe_visualization_name(value: Any) -> str:
    text = str(value)
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in text
    )[:160]


def normalize_spatial_distribution(tensor):
    import torch

    tensor = tensor.detach().float().cpu()
    return tensor / tensor.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def extract_scanflow_visual_distributions(
    payload: dict[str, Any],
    *,
    source: str,
    attention_layer_index: int,
    attention_fix_heads: int,
):
    import torch

    predictor_probs = payload.get("scanpath_predictor_probs")
    fixation_logits = payload.get("fixation_logits")
    visual_attn_maps = payload.get("visual_attn_maps")

    normalized_predictor_probs = None
    if torch.is_tensor(predictor_probs):
        predictor_probs = predictor_probs.detach().float().cpu()
        if predictor_probs.ndim == 2:
            predictor_probs = predictor_probs.unsqueeze(0)
        if predictor_probs.ndim != 3:
            raise ValueError(
                "Expected scanpath_predictor_probs [B,T,N], got "
                f"{tuple(predictor_probs.shape)}."
            )
        normalized_predictor_probs = normalize_spatial_distribution(
            predictor_probs
        )

    fixation_probs = None
    if torch.is_tensor(fixation_logits):
        fixation_logits = fixation_logits.detach().float().cpu()
        if fixation_logits.ndim == 2:
            fixation_logits = fixation_logits.unsqueeze(0)
        fixation_probs = torch.softmax(fixation_logits, dim=-1)

    attention_probs = None
    if torch.is_tensor(visual_attn_maps):
        visual_attn_maps = visual_attn_maps.detach().float().cpu()
        if visual_attn_maps.ndim == 4:
            visual_attn_maps = visual_attn_maps.unsqueeze(0)
        if visual_attn_maps.ndim != 5:
            raise ValueError(
                "Expected visual_attn_maps [B,L,T,H,N], got "
                f"{tuple(visual_attn_maps.shape)}."
            )

        layer_index = int(attention_layer_index)
        if layer_index < 0:
            layer_index += visual_attn_maps.size(1)
        if not 0 <= layer_index < visual_attn_maps.size(1):
            raise IndexError(
                f"Attention layer {attention_layer_index} is outside "
                f"{visual_attn_maps.size(1)} layers."
            )

        attention = visual_attn_maps[:, layer_index]
        if attention_fix_heads > 0:
            heads = min(int(attention_fix_heads), attention.size(2))
            attention = attention[:, :, :heads, :]

        attention_probs = attention.mean(dim=2)
        attention_probs = normalize_spatial_distribution(attention_probs)

    resolved_source = source
    if resolved_source == "auto":
        if normalized_predictor_probs is not None:
            resolved_source = "predictor"
        elif attention_probs is not None:
            resolved_source = "attention"
        else:
            resolved_source = "fixation"

    if resolved_source == "predictor":
        if normalized_predictor_probs is None:
            raise RuntimeError(
                "Plan 1 Parallel predictor visualization requested, but "
                "scanpath_predictor_probs were not captured."
            )
        primary = normalized_predictor_probs
    elif resolved_source == "attention":
        if attention_probs is None:
            raise RuntimeError(
                "Attention visualization requested, but no "
                "visual_attn_maps were captured."
            )
        primary = attention_probs
    elif resolved_source == "fixation":
        if fixation_probs is None:
            raise RuntimeError(
                "Fixation visualization requested, but no fixation_logits "
                "were captured."
            )
        primary = fixation_probs
    else:
        raise ValueError(f"Unsupported visualization source: {source}")

    return (
        resolved_source,
        primary,
        normalized_predictor_probs,
        fixation_probs,
        attention_probs,
    )


def pad_visualization_image(image, base_size: int = 1024):
    from PIL import ImageOps

    return ImageOps.pad(
        image.convert("RGB"),
        (base_size, base_size),
        color=(127, 127, 127),
    )


def patch_indices_to_xy(indices, grid_size: int, base_size: int = 1024):
    import numpy as np

    indices = np.asarray(indices)
    cell = base_size / grid_size
    rows = indices // grid_size
    cols = indices % grid_size
    return (cols + 0.5) * cell, (rows + 0.5) * cell


def save_distribution_heatmap_grid(
    *,
    distributions,
    title: str,
    output_path: Path,
    max_steps: int,
    cmap: str,
) -> None:
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    probs = distributions[0].detach().float().cpu()
    steps = min(int(probs.size(0)), int(max_steps))
    tokens = int(probs.size(-1))
    side = int(round(tokens ** 0.5))
    if side * side != tokens:
        raise ValueError(f"Spatial token count {tokens} is not square.")

    columns = min(5, max(1, steps))
    rows = int(math.ceil(steps / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.0 * columns, 3.0 * rows),
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)

    for step, axis in enumerate(axes):
        axis.axis("off")
        if step >= steps:
            continue
        axis.imshow(
            probs[step].reshape(side, side).numpy(),
            interpolation="bilinear",
            cmap=cmap,
        )
        axis.set_title(f"Step {step}", fontsize=9)

    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_distribution_overlay_grid(
    *,
    image,
    distributions,
    title: str,
    output_path: Path,
    max_steps: int,
    alpha: float,
    cmap: str,
) -> None:
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    probs = distributions[0].detach().float().cpu()
    steps = min(int(probs.size(0)), int(max_steps))
    tokens = int(probs.size(-1))
    side = int(round(tokens ** 0.5))
    if side * side != tokens:
        raise ValueError(f"Spatial token count {tokens} is not square.")

    image_1024 = pad_visualization_image(image)
    columns = min(5, max(1, steps))
    rows = int(math.ceil(steps / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.0 * columns, 3.0 * rows),
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)

    for step, axis in enumerate(axes):
        axis.axis("off")
        if step >= steps:
            continue
        axis.imshow(image_1024)
        axis.imshow(
            probs[step].reshape(side, side).numpy(),
            extent=(0, 1024, 1024, 0),
            interpolation="bilinear",
            alpha=alpha,
            cmap=cmap,
        )
        axis.set_title(f"Step {step}", fontsize=9)

    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_scanpath_overlay(
    *,
    image,
    primary_probs,
    primary_label: str,
    output_path: Path,
    max_steps: int,
    predictor_probs=None,
    fixation_probs=None,
    attention_probs=None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_1024 = pad_visualization_image(image)
    primary = primary_probs[0].detach().float().cpu()
    steps = min(int(primary.size(0)), int(max_steps))
    tokens = int(primary.size(-1))
    side = int(round(tokens ** 0.5))
    if side * side != tokens:
        raise ValueError(f"Spatial token count {tokens} is not square.")

    figure = plt.figure(figsize=(9, 9))
    plt.imshow(image_1024)

    plotted = set()

    def plot_probs(probs, label):
        if probs is None or label in plotted:
            return
        values = probs[0, :steps].detach().float().cpu()
        if values.size(-1) != tokens:
            return
        indices = values.argmax(dim=-1).numpy()
        x, y = patch_indices_to_xy(indices, side)
        plt.plot(x, y, marker="o", linewidth=2, label=label)
        for step in range(len(indices)):
            plt.text(x[step] + 5, y[step] + 5, str(step), fontsize=7)
        plotted.add(label)

    plot_probs(primary_probs, primary_label)
    plot_probs(predictor_probs, "Frozen predictor")
    plot_probs(attention_probs, "Attention")
    plot_probs(fixation_probs, "Fixation decoder")

    plt.title("ScanFlow predicted spatial trajectory")
    if plotted:
        plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def save_hidden_state_visualizations(
    *,
    scanpath_hidden,
    output_dir: Path,
    max_steps: int,
) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    if not torch.is_tensor(scanpath_hidden):
        return {}

    hidden = scanpath_hidden.detach().float().cpu()
    if hidden.ndim == 3:
        hidden = hidden[0]
    hidden = hidden[:max_steps]
    if hidden.ndim != 2 or hidden.size(0) == 0:
        return {}

    normalized = torch.nn.functional.normalize(
        hidden,
        dim=-1,
        eps=1e-12,
    )
    similarity = normalized @ normalized.transpose(0, 1)

    cosine_path = output_dir / "hidden_cosine_similarity.png"
    figure = plt.figure(figsize=(7, 6))
    image_handle = plt.imshow(
        similarity.numpy(),
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        origin="upper",
    )
    plt.colorbar(image_handle, label="Cosine similarity")
    plt.xlabel("Step")
    plt.ylabel("Step")
    plt.title("ScanFlow hidden-state cosine similarity")
    plt.tight_layout()
    plt.savefig(cosine_path, dpi=180)
    plt.close(figure)

    norms = torch.linalg.vector_norm(hidden, dim=-1)
    norm_path = output_dir / "hidden_norm_by_step.png"
    figure = plt.figure(figsize=(8, 4.5))
    plt.plot(range(hidden.size(0)), norms.numpy(), marker="o")
    plt.xlabel("Recurrent step")
    plt.ylabel("L2 norm")
    plt.title("ScanFlow hidden-state norm")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(norm_path, dpi=180)
    plt.close(figure)

    return {
        "hidden_cosine_similarity": str(cosine_path),
        "hidden_norm_by_step": str(norm_path),
    }


def render_scanflow_visualization(
    *,
    task: ScanFlowVisualizationTask,
    tensor_path: Path,
    output_dir: Path,
    source: str,
    attention_layer_index: int,
    attention_fix_heads: int,
    max_steps: int,
    alpha: float,
    cmap: str,
    rerun_prediction: str,
) -> dict[str, Any]:
    import torch

    payload = torch.load(
        tensor_path,
        map_location="cpu",
        weights_only=False,
    )

    (
        resolved_source,
        primary_probs,
        predictor_probs,
        fixation_probs,
        attention_probs,
    ) = extract_scanflow_visual_distributions(
        payload,
        source=source,
        attention_layer_index=attention_layer_index,
        attention_fix_heads=attention_fix_heads,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stable_tensor_path = output_dir / "scanflow_tensors.pt"
    torch.save(payload, stable_tensor_path)

    outputs: dict[str, Any] = {
        "tensor_file": str(stable_tensor_path),
        "prediction_source": resolved_source,
    }

    if predictor_probs is not None:
        path = output_dir / "predictor_heatmap_grid.png"
        save_distribution_heatmap_grid(
            distributions=predictor_probs,
            title="Plan 1 Parallel frozen predictor distributions",
            output_path=path,
            max_steps=max_steps,
            cmap=cmap,
        )
        outputs["predictor_heatmap_grid"] = str(path)

        path = output_dir / "predictor_overlay_grid.png"
        save_distribution_overlay_grid(
            image=task.image,
            distributions=predictor_probs,
            title="Plan 1 Parallel predictor overlays",
            output_path=path,
            max_steps=max_steps,
            alpha=alpha,
            cmap=cmap,
        )
        outputs["predictor_overlay_grid"] = str(path)

    if attention_probs is not None:
        path = output_dir / "attention_heatmap_grid.png"
        save_distribution_heatmap_grid(
            distributions=attention_probs,
            title="ScanFlow recurrent visual attention",
            output_path=path,
            max_steps=max_steps,
            cmap=cmap,
        )
        outputs["attention_heatmap_grid"] = str(path)

        path = output_dir / "attention_overlay_grid.png"
        save_distribution_overlay_grid(
            image=task.image,
            distributions=attention_probs,
            title="ScanFlow visual-attention overlays",
            output_path=path,
            max_steps=max_steps,
            alpha=alpha,
            cmap=cmap,
        )
        outputs["attention_overlay_grid"] = str(path)

    if fixation_probs is not None:
        path = output_dir / "fixation_heatmap_grid.png"
        save_distribution_heatmap_grid(
            distributions=fixation_probs,
            title="ScanFlow fixation-decoder predictions",
            output_path=path,
            max_steps=max_steps,
            cmap=cmap,
        )
        outputs["fixation_heatmap_grid"] = str(path)

        path = output_dir / "fixation_overlay_grid.png"
        save_distribution_overlay_grid(
            image=task.image,
            distributions=fixation_probs,
            title="ScanFlow fixation-decoder overlays",
            output_path=path,
            max_steps=max_steps,
            alpha=alpha,
            cmap=cmap,
        )
        outputs["fixation_overlay_grid"] = str(path)

    path = output_dir / "predicted_scanpath_overlay.png"
    save_scanpath_overlay(
        image=task.image,
        primary_probs=primary_probs,
        primary_label=resolved_source.capitalize(),
        output_path=path,
        max_steps=max_steps,
        predictor_probs=predictor_probs,
        fixation_probs=fixation_probs,
        attention_probs=attention_probs,
    )
    outputs["predicted_scanpath_overlay"] = str(path)

    outputs.update(
        save_hidden_state_visualizations(
            scanpath_hidden=(
                payload.get("scanpath_predictor_states")
                if payload.get("scanpath_predictor_states") is not None
                else payload.get("scanpath_hidden")
            ),
            output_dir=output_dir,
            max_steps=max_steps,
        )
    )

    manifest = {
        "dataset": task.dataset,
        "split": task.split,
        "sample_id": task.sample_id,
        "question_idx": task.question_idx,
        "question": task.question,
        "target_answer": task.answer,
        "original_prediction": task.prediction,
        "visualization_rerun_prediction": rerun_prediction,
        "prediction_matches_original": (
            clean_generated_text(rerun_prediction)
            == clean_generated_text(task.prediction)
        ),
        "metadata": task.metadata,
        "attention_layer_index": attention_layer_index,
        "attention_fix_heads": attention_fix_heads,
        "max_visualized_steps": max_steps,
        "outputs": outputs,
    }

    with (output_dir / "visualization_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    return manifest


def run_scanflow_visualization_pass(
    *,
    tasks: list[ScanFlowVisualizationTask],
    runner: DeepSeekOCR2VLLMRunner,
    output_root: Path,
    source: str,
    attention_layer_index: int,
    attention_fix_heads: int,
    max_steps: int,
    alpha: float,
    cmap: str,
    overwrite: bool,
) -> dict[str, int]:
    if not tasks:
        return {"requested": 0, "completed": 0, "errors": 0}

    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    completed = 0
    errors = 0

    print()
    print("=" * 100)
    print("POST-EVALUATION SCANFLOW VISUALIZATION PASS")
    print("=" * 100)
    print(
        f"Selected requests: {len(tasks)} | "
        f"source={source} | "
        f"attention_layer={attention_layer_index} | "
        f"attention_heads={attention_fix_heads}",
        flush=True,
    )

    for index, task in enumerate(tasks):
        sample_name = safe_visualization_name(task.sample_id)
        question_dir = (
            output_root
            / f"{task.dataset}_{task.split}"
            / f"{index:04d}_{sample_name}_q{task.question_idx}"
        )
        raw_dir = question_dir / "raw_runtime_tensors"

        capture_metadata = dict(task.metadata)
        capture_metadata.update(
            {
                "dataset": task.dataset,
                "split": task.split,
                "sample_id": task.sample_id,
                "question_idx": task.question_idx,
                "question_id": capture_metadata.get(
                    "question_id",
                    task.question_idx,
                ),
            }
        )

        try:
            rerun_prediction, tensor_path = (
                runner.generate_with_scanflow_capture(
                    request=task.request,
                    metadata=capture_metadata,
                    tensor_dir=raw_dir,
                )
            )

            if tensor_path is None:
                raise RuntimeError(
                    "The visualization re-run completed, but "
                    "scanflow_pro_deepseek_ocr2.py did not save a tensor payload. "
                    "Use tensor_parallel_size=1 / max_num_seqs=1 for the "
                    "visualization runner and verify the request metadata queue."
                )

            manifest = render_scanflow_visualization(
                task=task,
                tensor_path=tensor_path,
                output_dir=question_dir,
                source=source,
                attention_layer_index=attention_layer_index,
                attention_fix_heads=attention_fix_heads,
                max_steps=max_steps,
                alpha=alpha,
                cmap=cmap,
                rerun_prediction=rerun_prediction,
            )
            completed += 1
            print(
                f"[visualization {index + 1}/{len(tasks)}] "
                f"sample={task.sample_id} q={task.question_idx} "
                f"source={manifest['outputs']['prediction_source']} "
                f"prediction_match={manifest['prediction_matches_original']}",
                flush=True,
            )
        except Exception as exc:
            errors += 1
            question_dir.mkdir(parents=True, exist_ok=True)
            error_record = {
                "dataset": task.dataset,
                "split": task.split,
                "sample_id": task.sample_id,
                "question_idx": task.question_idx,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            with (question_dir / "visualization_error.json").open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(error_record, handle, indent=2, ensure_ascii=False)
            print(
                f"WARNING: visualization failed for sample={task.sample_id} "
                f"q={task.question_idx}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            try:
                task.image.close()
            except Exception:
                pass

    summary = {
        "requested": len(tasks),
        "completed": completed,
        "errors": errors,
    }
    with (output_root / "visualization_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    print(f"Visualization summary: {summary}", flush=True)
    return summary


def _payload_value(payload: dict[str, Any], logical_name: str):
    for key in V2_CAPTURE_ALIASES[logical_name]:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _squeeze_view_tensor(value, *, name: str):
    import torch

    if not torch.is_tensor(value):
        raise TypeError(f"V2 payload field {name!r} must be a tensor, got {type(value).__name__}.")
    value = value.detach().float().cpu()
    while value.ndim > 2 and value.size(0) == 1:
        value = value.squeeze(0)
    if name == "intensity":
        value = value.squeeze(-1)
        if value.ndim != 1:
            raise ValueError(f"Expected intensity [N] or [N,1], got {tuple(value.shape)}.")
    elif value.ndim != 2:
        raise ValueError(f"Expected {name} [N,D], got {tuple(value.shape)}.")
    return value


def extract_v2_views(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize either the preferred v2_views payload or a single-view payload."""
    import torch

    raw_views = payload.get("v2_views")
    if raw_views is None:
        raw_views = [payload]
    if not isinstance(raw_views, (list, tuple)):
        raise TypeError("v2_views must be a list of per-view dictionaries.")

    views: list[dict[str, Any]] = []
    for view_idx, raw in enumerate(raw_views):
        if not isinstance(raw, dict):
            raise TypeError(f"v2_views[{view_idx}] is not a dictionary.")
        original_raw = _payload_value(raw, "original")
        residual_raw = _payload_value(raw, "residual")
        intensity_raw = _payload_value(raw, "intensity")
        modified_raw = _payload_value(raw, "modified")
        missing = [
            name
            for name, value in (
                ("original", original_raw),
                ("residual", residual_raw),
                ("intensity", intensity_raw),
                ("modified", modified_raw),
            )
            if value is None
        ]
        if missing:
            raise KeyError(
                f"V2 capture view {view_idx} is missing {missing}. "
                "The vLLM runtime must save V, R, lambda, and V_modified."
            )
        original = _squeeze_view_tensor(original_raw, name="original")
        residual = _squeeze_view_tensor(residual_raw, name="residual")
        intensity = _squeeze_view_tensor(intensity_raw, name="intensity")
        modified = _squeeze_view_tensor(modified_raw, name="modified")
        if original.shape != residual.shape or original.shape != modified.shape:
            raise ValueError(
                f"V2 tensor mismatch: V={tuple(original.shape)}, R={tuple(residual.shape)}, "
                f"V'={tuple(modified.shape)}."
            )
        if intensity.numel() != original.size(0):
            raise ValueError(
                f"Intensity has {intensity.numel()} tokens but V has {original.size(0)}."
            )
        if (intensity < -1e-6).any() or (intensity > 1.0 + 1e-6).any():
            raise ValueError(
                "Dynamic V2 intensity is outside [0,1]; capture the post-sigmoid gate."
            )
        grid_shape = raw.get("grid_shape") or raw.get("view_grid_shape")
        if grid_shape is None:
            side = int(round(original.size(0) ** 0.5))
            if side * side != original.size(0):
                raise ValueError(
                    f"Cannot infer a square grid for N={original.size(0)}; save grid_shape."
                )
            grid_shape = (side, side)
        grid_shape = tuple(int(x) for x in grid_shape)
        if len(grid_shape) != 2 or grid_shape[0] * grid_shape[1] != original.size(0):
            raise ValueError(f"Invalid grid_shape={grid_shape} for N={original.size(0)}.")
        effective = intensity.unsqueeze(-1) * residual
        equation_max_error = float((modified - (original + effective)).abs().max().item())
        equation_tolerance = 5e-3 * (1.0 + float(modified.abs().max().item()))
        if equation_max_error > equation_tolerance:
            raise ValueError(
                "Captured tensors violate V' = V + lambda*R: "
                f"max_error={equation_max_error:.6g}, tolerance={equation_tolerance:.6g}."
            )
        if not torch.isfinite(original).all() or not torch.isfinite(modified).all():
            raise ValueError("Non-finite V2 visual tensor detected.")
        views.append(
            {
                "view_index": view_idx,
                "view_type": str(raw.get("view_type", "global" if grid_shape == (16, 16) else "local")),
                "grid_shape": grid_shape,
                "original": original,
                "residual": residual,
                "intensity": intensity,
                "effective": effective,
                "modified": modified,
                "projected_original": _payload_value(raw, "projected_original"),
                "projected_modified": _payload_value(raw, "projected_modified"),
                "equation_max_abs_error": equation_max_error,
            }
        )
    return views


def _tensor_stats(values, prefix: str) -> dict[str, float]:
    import torch

    values = values.detach().float().cpu().flatten()
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_q05": float(quantiles[0].item()),
        f"{prefix}_q25": float(quantiles[1].item()),
        f"{prefix}_median": float(quantiles[2].item()),
        f"{prefix}_q75": float(quantiles[3].item()),
        f"{prefix}_q95": float(quantiles[4].item()),
        f"{prefix}_max": float(values.max().item()),
    }


def compute_v2_view_metrics(view: dict[str, Any]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    original = view["original"]
    residual = view["residual"]
    intensity = view["intensity"]
    effective = view["effective"]
    modified = view["modified"]
    original_norm = original.norm(dim=-1)
    residual_norm = residual.norm(dim=-1)
    effective_norm = effective.norm(dim=-1)
    relative_update = effective_norm / original_norm.clamp_min(1e-12)
    cosine_change = 1.0 - F.cosine_similarity(original, modified, dim=-1, eps=1e-12)
    k = max(1, int(round(0.10 * intensity.numel())))
    positive_mass = intensity.clamp_min(0)
    topk_concentration = float(
        positive_mass.topk(k).values.sum().div(positive_mass.sum().clamp_min(1e-12)).item()
    )
    row: dict[str, Any] = {
        "view_index": view["view_index"],
        "view_type": view["view_type"],
        "grid_height": view["grid_shape"][0],
        "grid_width": view["grid_shape"][1],
        "num_visual_tokens": int(intensity.numel()),
        "embedding_dim": int(original.size(-1)),
        "lambda_cv": float(intensity.std(unbiased=False).div(intensity.mean().abs().clamp_min(1e-12)).item()),
        "lambda_top10pct_concentration": topk_concentration,
        "lambda_log10_mean": float(torch.log10(intensity.clamp_min(1e-12)).mean().item()),
        "equation_max_abs_error": view["equation_max_abs_error"],
    }
    row.update(_tensor_stats(intensity, "lambda"))
    row.update(_tensor_stats(residual_norm, "residual_norm"))
    row.update(_tensor_stats(effective_norm, "effective_update_norm"))
    row.update(_tensor_stats(relative_update, "relative_update"))
    row.update(_tensor_stats(cosine_change, "cosine_change"))

    projected_original = view.get("projected_original")
    projected_modified = view.get("projected_modified")
    if projected_original is not None and projected_modified is not None:
        projected_original = _squeeze_view_tensor(projected_original, name="projected_original")
        projected_modified = _squeeze_view_tensor(projected_modified, name="projected_modified")
        projected_delta = (projected_modified - projected_original).norm(dim=-1)
        projected_relative = projected_delta / projected_original.norm(dim=-1).clamp_min(1e-12)
        projected_cosine = 1.0 - F.cosine_similarity(
            projected_original, projected_modified, dim=-1, eps=1e-12
        )
        row.update(_tensor_stats(projected_relative, "projected_relative_update"))
        row.update(_tensor_stats(projected_cosine, "projected_cosine_change"))
    return row


def _map_arrays(view: dict[str, Any]) -> dict[str, Any]:
    import torch.nn.functional as F

    original = view["original"]
    modified = view["modified"]
    effective_norm = view["effective"].norm(dim=-1)
    relative = effective_norm / original.norm(dim=-1).clamp_min(1e-12)
    cosine = 1.0 - F.cosine_similarity(original, modified, dim=-1, eps=1e-12)
    return {
        "lambda": view["intensity"],
        "residual_norm": view["residual"].norm(dim=-1),
        "relative_update": relative,
        "cosine_change": cosine,
    }


def save_v2_change_panel(*, image, view: dict[str, Any], output_path: Path, alpha: float, cmap: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    maps = _map_arrays(view)
    grid_shape = view["grid_shape"]
    figure, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    axes[0].imshow(image)
    axes[0].set_title("Original chart")
    titles = {
        "lambda": r"Gate $\lambda_i(V,Q)$",
        "residual_norm": r"Residual $\|R_i\|_2$",
        "relative_update": r"Relative $\|\lambda_iR_i\|/\|V_i\|$",
        "cosine_change": r"$1-\cos(V_i,V'_i)$",
    }
    overlay_allowed = view["view_type"].lower() == "global" and grid_shape == (16, 16)
    background = np.asarray(pad_visualization_image(image))
    for axis, (name, values) in zip(axes[1:], maps.items()):
        grid = values.reshape(grid_shape).numpy()
        if overlay_allowed:
            axis.imshow(background)
            axis.imshow(
                grid,
                cmap=cmap,
                alpha=alpha,
                extent=(0, background.shape[1], background.shape[0], 0),
                interpolation="nearest",
            )
        else:
            axis.imshow(grid, cmap=cmap, interpolation="nearest")
        axis.set_title(titles[name])
        axis.set_axis_off()
    figure.suptitle(
        f"ScanFlow Pro V2 perception change — {view['view_type']} view {view['view_index']}",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _shared_pca_rgb(original, modified, grid_shape: tuple[int, int]):
    import torch

    combined = torch.cat([original, modified], dim=0).float()
    centered = combined - combined.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:3].T
    mins = projected.amin(dim=0, keepdim=True)
    maxs = projected.amax(dim=0, keepdim=True)
    rgb = (projected - mins) / (maxs - mins).clamp_min(1e-12)
    n = original.size(0)
    return (
        rgb[:n].reshape(*grid_shape, 3).numpy(),
        rgb[n:].reshape(*grid_shape, 3).numpy(),
    )


def save_v2_pca_panel(*, view: dict[str, Any], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    original_rgb, modified_rgb = _shared_pca_rgb(
        view["original"], view["modified"], view["grid_shape"]
    )
    difference = np.abs(modified_rgb - original_rgb)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, array, title in zip(
        axes,
        (original_rgb, modified_rgb, difference),
        ("Shared-PCA original V", "Shared-PCA modified V'", "Absolute PCA difference"),
    ):
        axis.imshow(array, interpolation="nearest")
        axis.set_title(title)
        axis.set_axis_off()
    figure.suptitle("Feature-space pseudo-RGB; colors are not reconstructed pixels", fontsize=11)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_v2_sample(
    *,
    task: ScanFlowVisualizationTask,
    payload: dict[str, Any],
    output_dir: Path,
    alpha: float,
    cmap: str,
    rerun_prediction: str,
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    stable_tensor_path = output_dir / "v2_selected_tensors.pt"
    torch.save(payload, stable_tensor_path)
    view_manifests = []
    for view in extract_v2_views(payload):
        prefix = f"view_{view['view_index']:02d}_{safe_visualization_name(view['view_type'])}"
        change_path = output_dir / f"{prefix}_perception_change.png"
        pca_path = output_dir / f"{prefix}_shared_pca.png"
        save_v2_change_panel(image=task.image, view=view, output_path=change_path, alpha=alpha, cmap=cmap)
        save_v2_pca_panel(view=view, output_path=pca_path)
        view_manifests.append(
            {
                "view_index": view["view_index"],
                "view_type": view["view_type"],
                "grid_shape": list(view["grid_shape"]),
                "metrics": compute_v2_view_metrics(view),
                "perception_change_panel": str(change_path),
                "shared_pca_panel": str(pca_path),
            }
        )
    manifest = {
        "dataset": task.dataset,
        "split": task.split,
        "sample_id": task.sample_id,
        "question_idx": task.question_idx,
        "question": task.question,
        "target_answer": task.answer,
        "original_prediction": task.prediction,
        "analysis_rerun_prediction": rerun_prediction,
        "prediction_matches_original": clean_generated_text(rerun_prediction) == clean_generated_text(task.prediction),
        "metadata": task.metadata,
        "tensor_file": str(stable_tensor_path),
        "views": view_manifests,
    }
    with (output_dir / "v2_analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest


def load_prediction_lookup(output_path: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not output_path.is_file():
        return lookup
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            lookup[str(record.get("sample_id"))] = record
    return lookup


def append_v2_metric(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_v2_analysis_dataset(
    *,
    dataset_name: str,
    split: str,
    datasets_root: Path,
    salchartqa_annotation_file: str | None,
    prediction_path: Path,
    runner: DeepSeekOCR2VLLMRunner,
    output_root: Path,
    max_samples: int | None,
    num_visualizations: int,
    alpha: float,
    cmap: str,
    overwrite: bool,
    continue_on_error: bool,
) -> dict[str, int]:
    import torch

    metrics_path = output_root / "v2_sample_metrics.jsonl"
    error_path = output_root / "v2_analysis_errors.jsonl"
    pair_path = output_root / "v2_same_image_question_pairs.jsonl"
    if overwrite:
        for path in (metrics_path, error_path, pair_path):
            if path.exists():
                path.unlink()
    existing_keys: set[str] = set()
    if metrics_path.is_file() and not overwrite:
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing_keys.add(json.loads(line)["analysis_key"])
    predictions = load_prediction_lookup(prediction_path)
    processed_samples = completed_questions = errors = visualized = 0
    for sample in iter_v2_evaluation_samples(
        dataset_name=dataset_name,
        split=split,
        datasets_root=datasets_root,
        batch_size=128,
        salchartqa_annotation_file=salchartqa_annotation_file,
    ):
        if max_samples is not None and processed_samples >= max_samples:
            break
        processed_samples += 1
        prediction_record = predictions.get(str(sample.sample_id))
        if prediction_record is None:
            continue
        image = load_sample_pil_image(sample)
        normalized_predictions = prediction_record.get("predictions", [])
        conversation_predictions: list[str | None] = [
            (str(value) if value is not None else None)
            for value in normalized_predictions
        ]
        if len(conversation_predictions) < len(sample.questions):
            conversation_predictions.extend(
                [None] * (len(sample.questions) - len(conversation_predictions))
            )
        question_gate_cache: list[tuple[int, Any]] = []
        try:
            for question_idx, question in enumerate(sample.questions):
                analysis_key = f"{dataset_name}:{split}:{sample.sample_id}:{question_idx}"
                if analysis_key in existing_keys:
                    continue
                prediction = str(normalized_predictions[question_idx]) if question_idx < len(normalized_predictions) else ""
                conversation = None
                if is_chartqapro_conversational_sample(dataset_name, sample.metadata):
                    conversation = build_chartqapro_conversation_history(
                        sample.questions, conversation_predictions, question_idx
                    )
                request = runner.build_request(
                    dataset_name=dataset_name,
                    image=image,
                    question=question,
                    metadata=sample.metadata,
                    conversation=conversation,
                )
                capture_metadata = dict(sample.metadata)
                capture_metadata.update(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "sample_id": sample.sample_id,
                        "question_idx": question_idx,
                        "analysis_key": analysis_key,
                    }
                )
                raw_dir = output_root / "raw_capture"
                try:
                    rerun_prediction, tensor_path = runner.generate_with_scanflow_capture(
                        request=request, metadata=capture_metadata, tensor_dir=raw_dir
                    )
                    conversation_predictions[question_idx] = normalize_prediction(
                        dataset_name, question, rerun_prediction, metadata=sample.metadata
                    )
                    if tensor_path is None:
                        raise RuntimeError("V2 runtime produced no analysis tensor payload.")
                    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
                    answer = str(sample.answers[question_idx]) if question_idx < len(sample.answers) else ""
                    base_record = {
                        "analysis_key": analysis_key,
                        "dataset": dataset_name,
                        "split": split,
                        "sample_id": sample.sample_id,
                        "question_idx": question_idx,
                        "question": question,
                        "answer": answer,
                        "prediction": prediction,
                        "analysis_rerun_prediction": rerun_prediction,
                        "prediction_matches_original": clean_generated_text(rerun_prediction) == clean_generated_text(prediction),
                        "is_correct_exact": prediction.strip().casefold() == answer.strip().casefold(),
                        "question_type": sample.metadata.get("question_type"),
                        "human_scanpath_length": sample.metadata.get("scanpath_length"),
                        "fixation_map_path": (
                            str(sample.metadata.get("fixation_map_path"))
                            if sample.metadata.get("fixation_map_path") is not None
                            else None
                        ),
                        "participant_id": sample.metadata.get("participant_id"),
                    }
                    captured_views = extract_v2_views(payload)
                    for view in captured_views:
                        row = dict(base_record)
                        row.update(compute_v2_view_metrics(view))
                        append_v2_metric(metrics_path, row)
                    preferred_view = next(
                        (view for view in captured_views if view["view_type"].lower() == "global"),
                        captured_views[0],
                    )
                    question_gate_cache.append(
                        (question_idx, preferred_view["intensity"].clone())
                    )
                    if visualized < num_visualizations:
                        task = ScanFlowVisualizationTask(
                            dataset=dataset_name,
                            split=split,
                            sample_id=sample.sample_id,
                            question_idx=question_idx,
                            question=question,
                            answer=answer,
                            prediction=prediction,
                            metadata=dict(sample.metadata),
                            request=request,
                            image=image.copy(),
                        )
                        question_dir = (
                            output_root / "visualizations" / f"{visualized:04d}_{safe_visualization_name(sample.sample_id)}_q{question_idx}"
                        )
                        render_v2_sample(
                            task=task,
                            payload=payload,
                            output_dir=question_dir,
                            alpha=alpha,
                            cmap=cmap,
                            rerun_prediction=rerun_prediction,
                        )
                        task.image.close()
                        visualized += 1
                    tensor_path.unlink(missing_ok=True)
                    completed_questions += 1
                except Exception as exc:
                    errors += 1
                    append_v2_metric(
                        error_path,
                        {
                            "analysis_key": analysis_key,
                            "dataset": dataset_name,
                            "split": split,
                            "sample_id": sample.sample_id,
                            "question_idx": question_idx,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        },
                    )
                    if not continue_on_error:
                        raise
            for left in range(len(question_gate_cache)):
                for right in range(left + 1, len(question_gate_cache)):
                    left_idx, left_gate = question_gate_cache[left]
                    right_idx, right_gate = question_gate_cache[right]
                    if left_gate.shape != right_gate.shape:
                        continue
                    centered_left = left_gate - left_gate.mean()
                    centered_right = right_gate - right_gate.mean()
                    correlation = float(
                        (centered_left * centered_right).sum().div(
                            centered_left.norm() * centered_right.norm() + 1e-12
                        ).item()
                    )
                    cosine = float(
                        (left_gate * right_gate).sum().div(
                            left_gate.norm() * right_gate.norm() + 1e-12
                        ).item()
                    )
                    append_v2_metric(
                        pair_path,
                        {
                            "dataset": dataset_name,
                            "split": split,
                            "sample_id": sample.sample_id,
                            "question_idx_a": left_idx,
                            "question_idx_b": right_idx,
                            "mean_lambda_abs_difference": float(
                                (left_gate.mean() - right_gate.mean()).abs().item()
                            ),
                            "map_mean_absolute_difference": float(
                                (left_gate - right_gate).abs().mean().item()
                            ),
                            "map_pearson_correlation": correlation,
                            "map_cosine_similarity": cosine,
                        },
                    )
        finally:
            image.close()
    try:
        (output_root / "raw_capture").rmdir()
    except OSError:
        pass
    return {
        "processed_samples": processed_samples,
        "completed_questions": completed_questions,
        "errors": errors,
        "visualized": visualized,
    }


def write_v2_aggregate_outputs(output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_paths = sorted((output_dir / "v2_analysis").glob("*/v2_sample_metrics.jsonl"))
    if not metric_paths:
        return
    rows = [
        json.loads(line)
        for metrics_path in metric_paths
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["view_type"])), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (dataset, view_type), items in sorted(groups.items()):
        summary_rows.append(
            {
                "dataset": dataset,
                "view_type": view_type,
                "num_question_views": len(items),
                "lambda_mean": sum(x["lambda_mean"] for x in items) / len(items),
                "lambda_spatial_std_mean": sum(x["lambda_std"] for x in items) / len(items),
                "lambda_top10pct_concentration_mean": sum(x["lambda_top10pct_concentration"] for x in items) / len(items),
                "relative_update_mean": sum(x["relative_update_mean"] for x in items) / len(items),
                "cosine_change_mean": sum(x["cosine_change_mean"] for x in items) / len(items),
                "exact_match_rate": sum(bool(x["is_correct_exact"]) for x in items) / len(items),
            }
        )
    summary_path = output_dir / "v2_benchmark_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    for metric, label in (
        ("lambda_mean", r"Sample mean $\lambda$"),
        ("lambda_std", r"Within-view spatial std of $\lambda$"),
        ("lambda_top10pct_concentration", r"Top-10% gate concentration"),
        ("relative_update_mean", r"Mean relative $\|\lambda R\|/\|V\|$"),
        ("cosine_change_mean", r"Mean $1-\cos(V,V')$"),
    ):
        labels = []
        values = []
        for (dataset, view_type), items in sorted(groups.items()):
            labels.append(f"{dataset}\n{view_type}")
            values.append([float(item[metric]) for item in items])
        figure, axis = plt.subplots(figsize=(max(8, 1.6 * len(labels)), 5))
        axis.violinplot(values, showmeans=True, showmedians=True)
        axis.set_xticks(range(1, len(labels) + 1), labels, rotation=25, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / f"v2_benchmark_{metric}_violin.png", dpi=180)
        plt.close(figure)


def finalize_sample_state(output_path: Path, state: SampleState) -> None:
    record = {
        "dataset": state.dataset,
        "split": state.split,
        "sample_id": state.sample_id,
        "questions": state.questions,
        "answers": state.answers,
        "predictions": state.predictions,
        "metadata": state.metadata,
        "elapsed_seconds": round(time.time() - state.started_at, 3),
    }
    question_errors = {str(idx): err for idx, err in enumerate(state.question_errors) if err}
    if question_errors:
        record["question_errors"] = question_errors
    jsonl_append(output_path, record)
    try:
        state.image.close()
    except Exception:
        pass


def build_pending_question(
    *,
    sample_key: str,
    question_idx: int,
    state: SampleState,
    runner: DeepSeekOCR2VLLMRunner,
) -> PendingQuestion:
    conversation = None
    if is_chartqapro_conversational_sample(state.dataset, state.metadata):
        conversation = build_chartqapro_conversation_history(
            state.questions,
            state.predictions,
            question_idx,
        )

    return PendingQuestion(
        sample_key=sample_key,
        question_idx=question_idx,
        request=runner.build_request(
            dataset_name=state.dataset,
            image=state.image,
            question=state.questions[question_idx],
            metadata=state.metadata,
            conversation=conversation,
        ),
    )


def enqueue_ready_questions(
    *,
    sample_key: str,
    state: SampleState,
    runner: DeepSeekOCR2VLLMRunner,
    pending_questions: list[PendingQuestion],
) -> None:
    if state.image is None:
        return

    if is_chartqapro_conversational_sample(state.dataset, state.metadata):
        if state.next_question_idx_to_schedule < len(state.questions):
            pending_questions.append(
                build_pending_question(
                    sample_key=sample_key,
                    question_idx=state.next_question_idx_to_schedule,
                    state=state,
                    runner=runner,
                )
            )
            state.next_question_idx_to_schedule += 1
        return

    while state.next_question_idx_to_schedule < len(state.questions):
        pending_questions.append(
            build_pending_question(
                sample_key=sample_key,
                question_idx=state.next_question_idx_to_schedule,
                state=state,
                runner=runner,
            )
        )
        state.next_question_idx_to_schedule += 1


def flush_question_batch(
    *,
    pending_questions: list[PendingQuestion],
    sample_states: dict[str, SampleState],
    runner: DeepSeekOCR2VLLMRunner,
    output_path: Path,
    continue_on_error: bool,
    visualization_collector: ScanFlowVisualizationCollector | None = None,
) -> tuple[int, int, list[PendingQuestion]]:
    if not pending_questions:
        return 0, 0, []

    written = 0
    errors = 0
    follow_up_questions: list[PendingQuestion] = []

    def update_success(task: PendingQuestion, prediction: str) -> None:
        nonlocal written
        state = sample_states[task.sample_key]
        state.predictions[task.question_idx] = normalize_prediction(
            state.dataset,
            state.questions[task.question_idx],
            prediction,
            metadata=state.metadata,
        )
        if visualization_collector is not None:
            visualization_collector.maybe_add(
                task=task,
                state=state,
            )
        enqueue_ready_questions(
            sample_key=task.sample_key,
            state=state,
            runner=runner,
            pending_questions=follow_up_questions,
        )
        state.remaining -= 1
        if state.remaining == 0:
            finalize_sample_state(output_path, state)
            del sample_states[task.sample_key]
            written += 1

    def update_error(task: PendingQuestion, exc: Exception) -> None:
        nonlocal written, errors
        errors += 1
        state = sample_states[task.sample_key]
        state.question_errors[task.question_idx] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        enqueue_ready_questions(
            sample_key=task.sample_key,
            state=state,
            runner=runner,
            pending_questions=follow_up_questions,
        )
        state.remaining -= 1
        if state.remaining == 0:
            finalize_sample_state(output_path, state)
            del sample_states[task.sample_key]
            written += 1

    try:
        predictions = runner.generate([task.request for task in pending_questions])
        for task, prediction in zip(pending_questions, predictions):
            update_success(task, prediction)
        return written, errors, follow_up_questions
    except Exception as batch_exc:
        if not continue_on_error:
            raise

        for task in pending_questions:
            try:
                prediction = runner.generate([task.request])[0]
                update_success(task, prediction)
            except Exception as single_exc:
                update_error(task, single_exc)
        return written, errors, follow_up_questions


def evaluate_dataset(
    *,
    dataset_name: str,
    split: str,
    datasets_root: Path,
    salchartqa_annotation_file: str | None,
    output_path: Path,
    runner: DeepSeekOCR2VLLMRunner,
    batch_size: int,
    overwrite: bool,
    max_samples: int | None,
    log_every: int,
    continue_on_error: bool,
    visualization_collector: ScanFlowVisualizationCollector | None = None,
) -> dict[str, int]:
    if overwrite and output_path.exists():
        output_path.unlink()

    completed_ids = load_existing_sample_ids(output_path)
    sample_states: dict[str, SampleState] = {}
    pending_questions: list[PendingQuestion] = []
    processed = 0
    skipped = 0
    errors = 0

    for sample in iter_v2_evaluation_samples(
        dataset_name=dataset_name,
        split=split,
        datasets_root=datasets_root,
        batch_size=128,
        salchartqa_annotation_file=salchartqa_annotation_file,
    ):
        sample_key = str(sample.sample_id)
        if sample_key in completed_ids:
            skipped += 1
            continue

        if max_samples is not None and processed >= max_samples:
            break

        if not sample.questions:
            state = SampleState(
                dataset=dataset_name,
                split=split,
                sample_id=sample.sample_id,
                questions=sample.questions,
                answers=sample.answers,
                predictions=[],
                question_errors=[],
                metadata=sample.metadata,
                image=None,
                started_at=time.time(),
                remaining=0,
            )
            finalize_sample_state(output_path, state)
            completed_ids.add(sample_key)
            processed += 1
            continue

        started_at = time.time()
        try:
            image = load_sample_pil_image(sample)
            state = SampleState(
                dataset=dataset_name,
                split=split,
                sample_id=sample.sample_id,
                questions=sample.questions,
                answers=sample.answers,
                predictions=[None] * len(sample.questions),
                question_errors=[None] * len(sample.questions),
                metadata=sample.metadata,
                image=image,
                started_at=started_at,
                remaining=len(sample.questions),
            )
            sample_states[sample_key] = state

            enqueue_ready_questions(
                sample_key=sample_key,
                state=state,
                runner=runner,
                pending_questions=pending_questions,
            )
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
                "question_errors": {"load": "".join(traceback.format_exception_only(type(exc), exc)).strip()},
                "elapsed_seconds": round(time.time() - started_at, 3),
            }
            jsonl_append(output_path, record)
            completed_ids.add(sample_key)
            processed += 1
            if not continue_on_error:
                raise
            continue

        processed += 1

        if len(pending_questions) >= batch_size:
            written_now, errors_now, follow_up_questions = flush_question_batch(
                pending_questions=pending_questions,
                sample_states=sample_states,
                runner=runner,
                output_path=output_path,
                continue_on_error=continue_on_error,
                visualization_collector=visualization_collector,
            )
            errors += errors_now
            pending_questions = follow_up_questions
            if processed % log_every == 0:
                print(
                    f"[{dataset_name}] processed_samples={processed} skipped={skipped} "
                    f"errors={errors} written={written_now}",
                    flush=True,
                )

    while pending_questions:
        written_now, errors_now, pending_questions = flush_question_batch(
            pending_questions=pending_questions,
            sample_states=sample_states,
            runner=runner,
            output_path=output_path,
            continue_on_error=continue_on_error,
            visualization_collector=visualization_collector,
        )
        errors += errors_now

    if sample_states:
        # Defensive fallback; normally states should already be flushed via remaining == 0.
        for state in list(sample_states.values()):
            finalize_sample_state(output_path, state)
        sample_states.clear()

    return {"processed": processed, "skipped": skipped, "errors": errors}


def parse_dataset_split_spec(spec: str) -> tuple[str, str]:
    dataset_name, separator, split = spec.partition(":")
    dataset_name = dataset_name.strip()
    split = split.strip()
    if not separator or not dataset_name or not split:
        raise ValueError(
            f"Invalid dataset split spec '{spec}'. Expected the format 'Dataset:split', "
            "for example 'ChartQA:val'."
        )
    return dataset_name, split


def build_eval_jobs(
    dataset_names: list[str],
    default_split: str,
    extra_dataset_splits: list[str],
) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for dataset_name in dataset_names:
        job = (dataset_name, default_split)
        if job not in seen:
            jobs.append(job)
            seen.add(job)

    for spec in extra_dataset_splits:
        job = parse_dataset_split_spec(spec)
        if job not in seen:
            jobs.append(job)
            seen.add(job)

    return jobs


def validate_prompt_template_arg(
    parser: argparse.ArgumentParser,
    option_name: str,
    template: str,
    *,
    required_fields: tuple[str, ...] = ("question",),
) -> None:
    if "<image>" not in template:
        parser.error(f"{option_name} must include '<image>'.")
    for field in required_fields:
        placeholder = f"{{{field}}}"
        if placeholder not in template:
            parser.error(f"{option_name} must include '{placeholder}'.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate dynamic ScanFlow Pro V2 with vLLM, then analyze its "
            "question-conditioned gate and reasoning-modified visual embeddings."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(dict.fromkeys([*DEFAULT_DATASET_NAMES, "SalChartQA"])),
        help="Datasets to evaluate. Default includes ChartBench, ChartQA, ChartQAPro, and SalChartQA.",
    )
    parser.add_argument("--split", default="test", help="Split to evaluate. Default: test.")
    parser.add_argument(
        "--salchartqa-annotation-file",
        default=None,
        help=(
            "SalChartQA JSONL annotation file, either absolute or relative to "
            "datasets/SalChartQA. Recommended for this analysis: "
            "splits/salchartqa_scanpath_test_one_per_question.jsonl."
        ),
    )
    parser.add_argument(
        "--extra-dataset-splits",
        nargs="*",
        default=[],
        help=(
            "Optional extra evaluation jobs in the form 'Dataset:split'. "
            "Example: --extra-dataset-splits ChartQA:val"
        ),
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help=f"Datasets root directory. Default: {DEFAULT_DATASETS_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store per-dataset prediction JSONL files.",
    )
    parser.add_argument(
        "--deepseek-vllm-dir",
        type=Path,
        default=DEFAULT_DEEPSEEK_VLLM_DIR,
        help="Path to DeepSeek-OCR2-vllm directory.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_V2_MODEL_NAME,
        help="HF model name or local model path.",
    )
    parser.add_argument(
        "--architecture",
        choices=("scanflow-pro-v2",),
        default="scanflow-pro-v2",
        help="Dynamic [V + lambda_i(V,Q)R; H] architecture.",
    )
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help=(
            "Prompt template for ChartQA and as a fallback for unknown ChartQAPro question types. "
            "Must include '<image>' and '{question}'."
        ),
    )
    parser.add_argument(
        "--chartqapro-factoid-prompt-template",
        default=DEFAULT_CHARTQAPRO_FACTOID_PROMPT_TEMPLATE,
        help="Prompt template for ChartQAPro factoid questions. Must include '<image>' and '{question}'.",
    )
    parser.add_argument(
        "--chartqapro-multi-choice-prompt-template",
        default=DEFAULT_CHARTQAPRO_MULTI_CHOICE_PROMPT_TEMPLATE,
        help=(
            "Prompt template for ChartQAPro multi-choice questions. Must include '<image>' "
            "and '{question}'."
        ),
    )
    parser.add_argument(
        "--chartqapro-hypothetical-prompt-template",
        default=DEFAULT_CHARTQAPRO_HYPOTHETICAL_PROMPT_TEMPLATE,
        help=(
            "Prompt template for ChartQAPro hypothetical questions. Must include '<image>' "
            "and '{question}'."
        ),
    )
    parser.add_argument(
        "--chartqapro-fact-checking-prompt-template",
        default=DEFAULT_CHARTQAPRO_FACT_CHECKING_PROMPT_TEMPLATE,
        help=(
            "Prompt template for ChartQAPro fact-checking questions. Must include '<image>' "
            "and '{question}'."
        ),
    )
    parser.add_argument(
        "--chartqapro-conversational-prompt-template",
        default=DEFAULT_CHARTQAPRO_CONVERSATIONAL_PROMPT_TEMPLATE,
        help=(
            "Prompt template for ChartQAPro conversational questions. Must include '<image>', "
            "'{question}', and '{conversation}'."
        ),
    )
    parser.add_argument(
        "--chartbench-statement-prompt-template",
        default=DEFAULT_CHARTBENCH_STATEMENT_PROMPT_TEMPLATE,
        help=(
            "Prompt template for ChartBench declarative statements. Must include '<image>' "
            "and '{question}'."
        ),
    )
    parser.add_argument(
        "--chartbench-value-prompt-template",
        default=DEFAULT_CHARTBENCH_VALUE_PROMPT_TEMPLATE,
        help=(
            "Prompt template for ChartBench value questions. Must include '<image>' "
            "and '{question}'."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value to set before model loading.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="vLLM dtype.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor_parallel_size. Default: 1.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="vLLM gpu_memory_utilization. Default: 0.75.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="vLLM max_model_len. Default: 8192.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=DEFAULT_MAX_NUM_SEQS,
        help=f"vLLM max_num_seqs. Default: {DEFAULT_MAX_NUM_SEQS}.",
    )
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=DEFAULT_MAX_NUM_SEQS,
        help=f"How many question-level requests to send in one llm.generate call. Default: {DEFAULT_MAX_NUM_SEQS}.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="vLLM block_size. Default: 256.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether to enable vLLM enforce_eager. Default: false, but text-query-residual "
            "checkpoints are forced to eager mode to avoid CUDA graph capture issues."
        ),
    )
    parser.add_argument(
        "--text-query-residual",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compatibility option retained for older launch scripts. Dynamic V2 "
            "always preserves its trained text-conditioned recurrent path."
        ),
    )
    parser.add_argument(
        "--disable-mm-preprocessor-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to disable vLLM multimodal preprocessor cache. Default: false.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max generated tokens per question. Default: 128.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0.0.",
    )
    parser.add_argument(
        "--use-no-repeat-ngram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DeepSeek's no-repeat-ngram logits processor. Default: false.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=20,
        help="ngram_size for no-repeat-ngram processor. Default: 20.",
    )
    parser.add_argument(
        "--no-repeat-window-size",
        type=int,
        default=90,
        help="window_size for no-repeat-ngram processor. Default: 90.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on processed samples per dataset for debugging.",
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=0,
        help=(
            "Number of analyzed questions whose full tensors, perception-change "
            "panels, and shared-PCA panels are retained. Default: 0."
        ),
    )
    parser.add_argument(
        "--v2-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run a single-request post-evaluation pass that records V2 gate and "
            "perception-change statistics. Default: enabled."
        ),
    )
    parser.add_argument(
        "--analysis-max-samples",
        type=int,
        default=None,
        help="Optional per-dataset sample cap for the V2 analysis pass.",
    )
    parser.add_argument(
        "--visualization-alpha",
        type=float,
        default=0.45,
        help="Opacity for heatmap overlays.",
    )
    parser.add_argument(
        "--visualization-cmap",
        default="jet",
        help="Matplotlib colormap for spatial heatmaps.",
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
        help="Continue after sample-level failures and record them to JSONL.",
    )
    return parser



def build_runner_from_args(
    args: argparse.Namespace,
    *,
    collect_visual_attention: bool,
    visualization_runner: bool = False,
) -> DeepSeekOCR2VLLMRunner:
    """
    Construct either the normal evaluation runner or the isolated visualization
    runner. The visualization runner is intentionally forced to single-request
    execution so runtime tensor metadata maps unambiguously to one question.
    """
    return DeepSeekOCR2VLLMRunner(
        deepseek_vllm_dir=args.deepseek_vllm_dir,
        model_name=args.model_name,
        prompt_template=args.prompt_template,
        chartbench_statement_prompt_template=args.chartbench_statement_prompt_template,
        chartbench_value_prompt_template=args.chartbench_value_prompt_template,
        chartqapro_factoid_prompt_template=args.chartqapro_factoid_prompt_template,
        chartqapro_multi_choice_prompt_template=args.chartqapro_multi_choice_prompt_template,
        chartqapro_hypothetical_prompt_template=args.chartqapro_hypothetical_prompt_template,
        chartqapro_fact_checking_prompt_template=args.chartqapro_fact_checking_prompt_template,
        chartqapro_conversational_prompt_template=args.chartqapro_conversational_prompt_template,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=(1 if visualization_runner else args.max_num_seqs),
        block_size=args.block_size,
        enforce_eager=True if visualization_runner else args.enforce_eager,
        text_query_residual=args.text_query_residual,
        disable_mm_preprocessor_cache=(
            True if visualization_runner else args.disable_mm_preprocessor_cache
        ),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_no_repeat_ngram=args.use_no_repeat_ngram,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        no_repeat_window_size=args.no_repeat_window_size,
        architecture=args.architecture,
        collect_visual_attention=collect_visual_attention,
    )

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    # Accept either a checkpoint directory or the parent ms-swift vX-... run
    # directory supplied by the user.
    args.model_name = resolve_latest_checkpoint_dir(args.model_name)

    validate_prompt_template_arg(parser, "--prompt-template", args.prompt_template)
    validate_prompt_template_arg(
        parser,
        "--chartqapro-factoid-prompt-template",
        args.chartqapro_factoid_prompt_template,
    )
    validate_prompt_template_arg(
        parser,
        "--chartqapro-multi-choice-prompt-template",
        args.chartqapro_multi_choice_prompt_template,
    )
    validate_prompt_template_arg(
        parser,
        "--chartqapro-hypothetical-prompt-template",
        args.chartqapro_hypothetical_prompt_template,
    )
    validate_prompt_template_arg(
        parser,
        "--chartqapro-fact-checking-prompt-template",
        args.chartqapro_fact_checking_prompt_template,
    )
    validate_prompt_template_arg(
        parser,
        "--chartqapro-conversational-prompt-template",
        args.chartqapro_conversational_prompt_template,
        required_fields=("question", "conversation"),
    )
    validate_prompt_template_arg(
        parser,
        "--chartbench-statement-prompt-template",
        args.chartbench_statement_prompt_template,
    )
    validate_prompt_template_arg(
        parser,
        "--chartbench-value-prompt-template",
        args.chartbench_value_prompt_template,
    )

    if args.request_batch_size <= 0:
        parser.error("--request-batch-size must be positive.")
    if args.log_every <= 0:
        parser.error("--log-every must be positive.")
    if args.num_visualizations < 0:
        parser.error("--num-visualizations must be nonnegative.")
    if not 0.0 <= args.visualization_alpha <= 1.0:
        parser.error("--visualization-alpha must be in [0, 1].")
    try:
        eval_jobs = build_eval_jobs(args.datasets, args.split, args.extra_dataset_splits)
    except ValueError as exc:
        parser.error(str(exc))

    has_salchartqa = any(
        name.strip().lower().replace("-", "").replace("_", "")
        in {"salchartqa", "salchartqascanpath"}
        for name, _ in eval_jobs
    )
    if has_salchartqa and args.salchartqa_annotation_file:
        annotation_path = Path(args.salchartqa_annotation_file).expanduser()
        if not annotation_path.is_absolute():
            annotation_path = args.datasets_root / "SalChartQA" / annotation_path
        if not annotation_path.is_file():
            parser.error(f"SalChartQA annotation file not found: {annotation_path}")
        print(f"SalChartQA annotation: {annotation_path}", flush=True)
    elif has_salchartqa:
        print(
            "WARNING: SalChartQA is enabled without --salchartqa-annotation-file; "
            "the dataloader's default candidate order will be used.",
            file=sys.stderr,
            flush=True,
        )

    # Fast batched answer generation remains separate from internal-state capture.
    runner = build_runner_from_args(
        args,
        collect_visual_attention=False,
        visualization_runner=False,
    )

    overall_exit_code = 0
    try:
        for dataset_name, split_name in eval_jobs:
            output_path = build_output_path(args.output_dir, dataset_name, split_name)
            print(f"=== Evaluating {dataset_name} ({split_name}) with vLLM ===", flush=True)
            print(f"output_file: {output_path}", flush=True)
            try:
                stats = evaluate_dataset(
                    dataset_name=dataset_name,
                    split=split_name,
                    datasets_root=args.datasets_root,
                    salchartqa_annotation_file=args.salchartqa_annotation_file,
                    output_path=output_path,
                    runner=runner,
                    batch_size=args.request_batch_size,
                    overwrite=args.overwrite,
                    max_samples=args.max_samples,
                    log_every=args.log_every,
                    continue_on_error=args.continue_on_error,
                    visualization_collector=None,
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

        if args.v2_analysis:
            runner.shutdown()
            analysis_runner = build_runner_from_args(
                args,
                # This flag now means the V2 runtime capture contract is required.
                collect_visual_attention=True,
                visualization_runner=True,
            )
            try:
                remaining_visualizations = args.num_visualizations
                for dataset_name, split_name in eval_jobs:
                    prediction_path = build_output_path(args.output_dir, dataset_name, split_name)
                    analysis_root = args.output_dir / "v2_analysis" / f"{dataset_name}_{split_name}"
                    print(
                        f"=== V2 gate/perception analysis: {dataset_name} ({split_name}) ===",
                        flush=True,
                    )
                    stats = run_v2_analysis_dataset(
                        dataset_name=dataset_name,
                        split=split_name,
                        datasets_root=args.datasets_root,
                        salchartqa_annotation_file=args.salchartqa_annotation_file,
                        prediction_path=prediction_path,
                        runner=analysis_runner,
                        output_root=analysis_root,
                        max_samples=(
                            args.analysis_max_samples
                            if args.analysis_max_samples is not None
                            else args.max_samples
                        ),
                        num_visualizations=remaining_visualizations,
                        alpha=args.visualization_alpha,
                        cmap=args.visualization_cmap,
                        overwrite=args.overwrite,
                        continue_on_error=args.continue_on_error,
                    )
                    remaining_visualizations = max(
                        0, remaining_visualizations - stats["visualized"]
                    )
                    print(f"V2 analysis finished: {stats}", flush=True)
                write_v2_aggregate_outputs(args.output_dir)
            except Exception as exc:
                print(
                    f"WARNING: V2 post-evaluation analysis failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    traceback.format_exc(),
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                analysis_runner.shutdown()

        return overall_exit_code
    finally:
        runner.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
