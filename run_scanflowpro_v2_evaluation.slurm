#!/bin/bash
#SBATCH --job-name=scanflow_pro_v2_analysis
#SBATCH --account=def-mageed
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/krisxu/scratch/2026_summer_research_project/CueFlow/logs/%x-%j.out
#SBATCH --error=/home/krisxu/scratch/2026_summer_research_project/CueFlow/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="/home/krisxu/scratch/2026_summer_research_project/CueFlow"
VENV_DIR="/home/krisxu/scratch/venvs/cueflow311"

EVAL_SCRIPT="${PROJECT_DIR}/run_deepseek_ocr2_eval_vllm_scanflow_pro_v2_analysis.py"
DEEPSEEK_VLLM_DIR="${PROJECT_DIR}/DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm"

# Override at submission time after the downstream V2 LoRA is merged:
# sbatch --export=ALL,SCANFLOW_V2_EVAL_MODEL_PATH=/absolute/model/path \
#   eval_scanflow_pro_v2_analysis_fir.slurm
MODEL_PATH="${SCANFLOW_V2_EVAL_MODEL_PATH:-${PROJECT_DIR}/output/scanflow_pro_v2_dynamic_intensity_attention/v0-20260902-151808/checkpoint-3600}"

DATASETS_ROOT="${PROJECT_DIR}/datasets"
OUTPUT_DIR="${SCANFLOW_V2_EVAL_OUTPUT_DIR:-${PROJECT_DIR}/eval_outputs/vllm/scanflow_pro_v2_dynamic_analysis/checkpoint-3600}"

# Override with, for example:
#   --export=ALL,SCANFLOW_V2_DATASETS="SalChartQA"
# or split a long run across jobs by selecting one dataset per submission.
DATASET_LIST="${SCANFLOW_V2_DATASETS:-ChartQA}"
read -r -a DATASETS <<< "${DATASET_LIST}"
SPLIT="test"
SALCHARTQA_ANNOTATION_FILE="splits/salchartqa_scanpath_test_one_per_question.jsonl"

# Empty means the complete split. For a smoke test, submit with:
# sbatch --export=ALL,SCANFLOW_V2_MAX_SAMPLES=20 ...
MAX_SAMPLES="${SCANFLOW_V2_MAX_SAMPLES:-}"
ANALYSIS_MAX_SAMPLES="${SCANFLOW_V2_ANALYSIS_MAX_SAMPLES:-}"
NUM_VISUALIZATIONS="${SCANFLOW_V2_NUM_VISUALIZATIONS:-20}"
OVERWRITE="${SCANFLOW_V2_OVERWRITE:-0}"

TENSOR_PARALLEL_SIZE=1
MAX_NUM_SEQS=1
REQUEST_BATCH_SIZE=1
GPU_MEMORY_UTILIZATION=0.75
MAX_MODEL_LEN=8192
BLOCK_SIZE=256
MAX_TOKENS=128
TEMPERATURE=0.0

mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"

echo "============================================================"
echo "ScanFlow Pro V2 dynamic-gate and perception analysis"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"

# ============================================================
# Fir software environment
# ============================================================

module --force purge
module load StdEnv/2023
module load gcc
module load python/3.11
module load cuda/12.2
module load arrow/21.0.0
module load opencv/4.12.0

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/ms-swift:${DEEPSEEK_VLLM_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

export HF_HOME="${PROJECT_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules_scanflow_pro_v2_vllm_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"

mkdir -p \
    "${HF_HOME}" \
    "${HF_DATASETS_CACHE}" \
    "${HF_MODULES_CACHE}" \
    "${XDG_CACHE_HOME}" \
    "${TORCH_HOME}" \
    "${TRITON_CACHE_DIR}"

# ============================================================
# Dynamic ScanFlow Pro V2 inference architecture
# ============================================================

export SCANFLOW_PLAN2=0
export SCANFLOW_DYNAMIC_CAUSAL_TOKENS=0
export SCANFLOW_RESET_NEW_PARAMS=0
export SCANFLOW_V2_RESET_NEW_PARAMS=0
export SCANFLOW_V2_INFERENCE_STEPS=32

# Legacy template switch used for the shared N+T image-placeholder layout.
export SCANFLOW_PRO_V4=1

export DEEPSEEK_OCR2_ENABLE_TEXT_QUERY_RESIDUAL=1
export DEEPSEEK_OCR2_TEXT_QUERY_RESIDUAL_ONLY=0
export DEEPSEEK_OCR2_SCANPATH_START_SOURCE=visual_mean

# Inference uses no auxiliary objectives. The V2 information flow remains on.
export DEEPSEEK_OCR2_ENABLE_ATTENTION_SUPERVISION=0
export DEEPSEEK_OCR2_ATTENTION_LOSS_WEIGHT=0.0
export DEEPSEEK_OCR2_ATTENTION_DIVERSITY_LOSS_WEIGHT=0.0
export DEEPSEEK_OCR2_FIXATION_LOSS_WEIGHT=0.0
export DEEPSEEK_OCR2_DIVERSITY_LOSS_WEIGHT=0.0
export SCANFLOW_PLAN1_PREDICTOR_LOSS_WEIGHT=0.0

export DEEPSEEK_OCR2_COMPUTE_ATTENTION_DIAGNOSTICS=0
export DEEPSEEK_OCR2_COMPUTE_FIXATION_DIAGNOSTICS=0
export DEEPSEEK_OCR2_COMPUTE_ATTENTION_DIVERSITY_DIAGNOSTICS=0
export SCANFLOW_LOG_ATTENTION_OUTPUTS=0
export SCANFLOW_LOG_FIXATION_OUTPUTS=0
export SCANFLOW_DEBUG_TENSORS=0

# The evaluator enables these only during the single-request analysis pass.
unset SCANFLOW_V2_SAVE_ANALYSIS || true
unset SCANFLOW_V2_ANALYSIS_DIR || true
unset SCANFLOW_SAVE_ONLY_WITH_METADATA || true

export CROP_MODE=1
export BASE_SIZE=1024
export IMAGE_SIZE=768

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0

# ============================================================
# File and architecture preflight
# ============================================================

SCANFLOW_ENCODER="${DEEPSEEK_VLLM_DIR}/scanflow_pro_deepencoder.py"
SCANFLOW_WRAPPER="${DEEPSEEK_VLLM_DIR}/scanflow_pro_deepseek_ocr2.py"

for REQUIRED_FILE in "${EVAL_SCRIPT}" "${SCANFLOW_ENCODER}" "${SCANFLOW_WRAPPER}"; do
    if [[ ! -f "${REQUIRED_FILE}" ]]; then
        echo "ERROR: required file missing: ${REQUIRED_FILE}" >&2
        exit 1
    fi
done

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: V2 model directory not found: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "ERROR: config.json missing from model directory: ${MODEL_PATH}" >&2
    exit 1
fi

for DATASET_NAME in "${DATASETS[@]}"; do
    if [[ ! -d "${DATASETS_ROOT}/${DATASET_NAME}" ]]; then
        echo "ERROR: dataset directory missing: ${DATASETS_ROOT}/${DATASET_NAME}" >&2
        exit 1
    fi
done

if [[ " ${DATASETS[*]} " == *" SalChartQA "* ]]; then
    SALCHARTQA_ANNOTATION_PATH="${DATASETS_ROOT}/SalChartQA/${SALCHARTQA_ANNOTATION_FILE}"
    if [[ ! -f "${SALCHARTQA_ANNOTATION_PATH}" ]]; then
        echo "ERROR: SalChartQA annotation file missing: ${SALCHARTQA_ANNOTATION_PATH}" >&2
        exit 1
    fi
fi

for MARKER in \
    scanflow_v2_residual_attn \
    scanflow_v2_intensity_mlp \
    scanflow_v2_intensity_visual_norm \
    scanflow_v2_intensity_question_norm
do
    if ! grep -q "${MARKER}" "${SCANFLOW_ENCODER}"; then
        echo "ERROR: V2 encoder lacks marker ${MARKER}" >&2
        exit 1
    fi
done

if ! grep -Fq 'combined = torch.cat([visual_tokens, causal_inputs], dim=1)' "${SCANFLOW_ENCODER}"; then
    echo "ERROR: encoder does not preserve immutable-V recurrent generation." >&2
    exit 1
fi

for MARKER in \
    scanflow_pro_deepencoder \
    SCANFLOW_V2_SAVE_ANALYSIS \
    SCANFLOW_V2_ANALYSIS_DIR \
    v2_original_visual \
    v2_reasoning_residual \
    v2_intensity \
    v2_modified_visual \
    v2_views
do
    if ! grep -q "${MARKER}" "${SCANFLOW_WRAPPER}"; then
        echo "ERROR: V2 wrapper lacks marker ${MARKER}" >&2
        exit 1
    fi
done

if ! grep -q 'global_views_tokens = h \* w + scanflow_steps' "${SCANFLOW_WRAPPER}"; then
    echo "ERROR: wrapper does not reserve N+T global tokens." >&2
    exit 1
fi

if ! grep -q 'ScanflowProDeepseekOCR2ForCausalLM' "${EVAL_SCRIPT}"; then
    echo "ERROR: evaluator lacks the renamed ScanFlow Pro vLLM registry." >&2
    exit 1
fi

if ! grep -q 'from scanflow_pro_deepseek_ocr2 import' "${EVAL_SCRIPT}"; then
    echo "ERROR: evaluator does not import the renamed V2 wrapper." >&2
    exit 1
fi

python -m py_compile "${EVAL_SCRIPT}" "${SCANFLOW_ENCODER}" "${SCANFLOW_WRAPPER}"

# ============================================================
# Package and GPU preflight
# ============================================================

python - <<'PY'
import sys
import torch
import tokenizers
import transformers

print("Python:", sys.executable)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("tokenizers:", tokenizers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible.")
print("GPU 0:", torch.cuda.get_device_name(0))
if transformers.__version__.split("+")[0] != "4.46.3":
    raise RuntimeError("Expected transformers==4.46.3")
if tokenizers.__version__.split("+")[0] != "0.20.3":
    raise RuntimeError("Expected tokenizers==0.20.3")
PY

nvidia-smi \
    --query-gpu=index,uuid,name,memory.total,memory.used,memory.free \
    --format=csv

# ============================================================
# Build evaluation command
# ============================================================

CMD=(
    python "${EVAL_SCRIPT}"
    --architecture scanflow-pro-v2
    --datasets "${DATASETS[@]}"
    --split "${SPLIT}"
    --salchartqa-annotation-file "${SALCHARTQA_ANNOTATION_FILE}"
    --datasets-root "${DATASETS_ROOT}"
    --deepseek-vllm-dir "${DEEPSEEK_VLLM_DIR}"
    --model-name "${MODEL_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --dtype bfloat16
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --request-batch-size "${REQUEST_BATCH_SIZE}"
    --block-size "${BLOCK_SIZE}"
    --enforce-eager
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --v2-analysis
    --num-visualizations "${NUM_VISUALIZATIONS}"
    --visualization-alpha 0.45
    --visualization-cmap jet
    --log-every 10
    --continue-on-error
)

if [[ -n "${MAX_SAMPLES}" ]]; then
    CMD+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ -n "${ANALYSIS_MAX_SAMPLES}" ]]; then
    CMD+=(--analysis-max-samples "${ANALYSIS_MAX_SAMPLES}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
    CMD+=(--overwrite)
fi

echo "============================================================"
echo "Evaluation configuration"
echo "Datasets: ${DATASETS[*]}"
echo "Split: ${SPLIT}"
echo "SalChartQA annotation: ${SALCHARTQA_ANNOTATION_FILE}"
echo "Model: ${MODEL_PATH}"
echo "Inference steps per view: ${SCANFLOW_V2_INFERENCE_STEPS}"
echo "Analysis max samples: ${ANALYSIS_MAX_SAMPLES:-full}"
echo "Full-tensor visualizations: ${NUM_VISUALIZATIONS}"
echo "Overwrite: ${OVERWRITE}"
echo "Command:"
printf ' %q' "${CMD[@]}"
echo
echo "============================================================"

"${CMD[@]}"

echo "============================================================"
echo "Evaluation completed: $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "============================================================"
