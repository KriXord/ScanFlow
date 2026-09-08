#!/bin/bash
#SBATCH --job-name=scanflow_v2_dynamic
#SBATCH --account=def-mageed
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/krisxu/scratch/2026_summer_research_project/CueFlow/logs/%x-%j.out
#SBATCH --error=/home/krisxu/scratch/2026_summer_research_project/CueFlow/logs/%x-%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# ============================================================
# Paths
# ============================================================

PROJECT_DIR="/home/krisxu/scratch/2026_summer_research_project/CueFlow"
VENV_DIR="/home/krisxu/scratch/venvs/cueflow311"
SWIFT_DIR="${PROJECT_DIR}/ms-swift"
SWIFT_BASE_TEMPLATE="${SWIFT_DIR}/swift/template/base.py"
SWIFT_DEEPSEEK_TEMPLATE="${SWIFT_DIR}/swift/template/templates/deepseek.py"

MODEL_PATH="/home/krisxu/scratch/2026_summer_research_project/CueFlow/output/scanflow_plan1_recurrent_attention/v8-20260818-033505/scanflowpro_v2"

DATASET_PATH="${PROJECT_DIR}/datasets/SalChartQA/splits/salchartqa_scanpath_train_correct_swift_one_per_question.jsonl"

OUTPUT_DIR="${PROJECT_DIR}/output/scanflow_pro_v2_dynamic_intensity_attention"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"

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

export PYTHONPATH="${SWIFT_DIR}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

# One Fir node with four full H100-80GB GPUs.
# The explicit h100 model selector avoids requesting an H100 MIG profile.
#
# NOTE:
# This is still DDP/data parallelism. Each GPU holds one full model replica;
# the four GPUs do NOT combine into one 320-GB model-memory pool.
export NPROC_PER_NODE=4

# ============================================================
# Writable cache paths
# ============================================================
# Keep caches out of /home/krisxu/.cache on Fir.

export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"

export MODELSCOPE_CACHE="${XDG_CACHE_HOME}/modelscope"
export MODELSCOPE_HOME="${XDG_CACHE_HOME}/modelscope"
export MODELSCOPE_MODULES_CACHE="${XDG_CACHE_HOME}/modelscope/modules"
export MODELSCOPE_INDEX_FILE="${XDG_CACHE_HOME}/modelscope/ast_indexer"

export HF_HOME="${PROJECT_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules_scanflow_pro_v2_${SLURM_JOB_ID}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"

mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${MODELSCOPE_CACHE}" \
  "${MODELSCOPE_MODULES_CACHE}" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_MODULES_CACHE}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${MPLCONFIGDIR}"

# ============================================================
# ScanFlow Pro V2 template / architecture
# ============================================================

# V2 returns [V + lambda_i(V,Q)*R; H] for each encoded view, so the patched
# ms-swift template must reserve N_visual + T <image> positions per view.
export SCANFLOW_PRO_V2=1
export SCANFLOW_PRO_V4=1
export SCANFLOW_DYNAMIC_CAUSAL_TOKENS=1

# IMPORTANT: SCANFLOW_PRO_V4 is retained only as the current ms-swift template's
# legacy name for its N_visual+T placeholder branch. It does not enable V4 in
# deepencoderv2.py. The model-folder code and SCANFLOW_PRO_V2 identify the actual
# architecture used by this run.

# Do not identify this run as old Plan 2. The modified template keeps
# SCANFLOW_PLAN2 only as a legacy alias.
export SCANFLOW_PLAN2=0

# Preserve the recurrent Plan-1 information flow exactly:
#   h_t = Qwen(V, h_1, ..., h_(t-1), seed_t)[-1]
# Only after H is complete:
#   R = CrossAttention(query=V, key=H, value=H)
#   lambda_i(V,Q) = sigmoid(F_I([v_i, pool(V), pool(Q)]))
#   X_LLM = [V + lambda(V,Q)*R; H]
# There is no stepwise visual-memory update and no residual feedback into H.
unset SCANFLOW_PRO_USE_ORIGINAL_VISUAL_IN_FM

# ============================================================
# Attention supervision
# ============================================================

# Human scanpaths supervise recurrent Qwen self-attention from each newly
# generated h_t state to global visual tokens.
export DEEPSEEK_OCR2_ENABLE_ATTENTION_SUPERVISION=1
export DEEPSEEK_OCR2_ATTENTION_LOSS_WEIGHT=5.0

# New middle-layer supervision:
#
# Qwen has 24 decoder layers (0-based indices 0..23).
# Supervise the four central layers:
#     10, 11, 12, 13
#
# Within EACH selected layer:
#   - average over ALL 14 query-attention heads
#   - align that layer-level visual distribution to the temporal fixation target
#
# The four per-layer fixation losses are then averaged.
#
# This variable is technically optional because the modified deepencoderv2.py
# automatically selects the middle four layers for a 24-layer encoder, but we
# set it explicitly here for reproducibility and clear experiment logging.
export DEEPSEEK_OCR2_ATTENTION_LAYER_INDICES="10,11,12,13"

# Obsolete controls from the previous single-layer / first-two-head design.
unset DEEPSEEK_OCR2_ATTENTION_LAYER_INDEX
unset DEEPSEEK_OCR2_ATTENTION_FIX_HEADS

# Preserve all learned recurrent Plan-1 parameters from MODEL_PATH.
export SCANFLOW_RESET_NEW_PARAMS=0

# Reset only the new V2 global-residual cross-attention and dynamic reasoning-
# intensity head after loading the recurrent Plan-1 checkpoint. Every tokenwise
# lambda_i is initialized exactly to 1e-3 for a checkpoint-safe transition.
# Set this to 0 when resuming from a checkpoint already trained as V2.
export SCANFLOW_V2_RESET_NEW_PARAMS=1
export SCANFLOW_V2_INITIAL_INTENSITY=0.001
unset SCANFLOW_PRO_RESET_NEW_PARAMS

# The recurrent Plan-1 model does NOT use the old decoded-fixation,
# fixation-diversity, routing, residual, or attention-diversity losses.
unset DEEPSEEK_OCR2_FIXATION_LOSS_WEIGHT
unset DEEPSEEK_OCR2_DIVERSITY_LOSS_WEIGHT
unset DEEPSEEK_OCR2_ATTENTION_DIVERSITY_LOSS_WEIGHT
unset DEEPSEEK_OCR2_ENABLE_TEXT_QUERY_RESIDUAL
unset DEEPSEEK_OCR2_TEXT_QUERY_RESIDUAL_ONLY
unset DEEPSEEK_OCR2_SCANPATH_START_SOURCE
unset SCANFLOW_INIT_USE_TEXT_SUMMARY
unset SCANFLOW_ROUTING_TEMPERATURE
unset SCANFLOW_DETACH_ROUTING_KEYS

# ============================================================
# Diagnostics
# ============================================================

export SCANFLOW_LOG_ATTENTION_OUTPUTS=1
export SCANFLOW_DIAGNOSTIC_LOG_INTERVAL=10

export DEEPSEEK_OCR2_LOG_GRADIENTS=1
export DEEPSEEK_OCR2_GRAD_LOG_INTERVAL=10
export DEEPSEEK_OCR2_LOG_NAMED_GRADIENTS=1
export DEEPSEEK_OCR2_NAMED_GRADIENT_LIMIT=80

export DEEPSEEK_OCR2_LOG_PARAMETER_UPDATES=1
export DEEPSEEK_OCR2_UPDATE_LOG_INTERVAL=10
export DEEPSEEK_OCR2_LOG_LOSSES=1

# Restrict diagnostic reporting to the recurrent visual causal encoder, the
# new V2 residual/intensity branch, and the question-conditioning projection.
export DEEPSEEK_OCR2_UPDATE_PARAM_FILTER="qwen2_model.model.model,qwen2_model.model.text_cross_norms,qwen2_model.model.text_cross_attn_layers,qwen2_model.model.scanflow_v2_visual_norm,qwen2_model.model.scanflow_v2_reasoning_norm,qwen2_model.model.scanflow_v2_residual_attn,qwen2_model.model.scanflow_v2_intensity_visual_norm,qwen2_model.model.scanflow_v2_intensity_question_norm,qwen2_model.model.scanflow_v2_intensity_mlp,qwen2_model.text_context_proj"

# ============================================================
# Image preprocessing
# ============================================================

# Keep crop mode enabled for the real run.
# If this still OOMs, the cleanest diagnostic is:
#   export CROP_MODE=0
# which removes recurrent local-crop passes and tests global recurrence alone.
export CROP_MODE=1
export BASE_SIZE=1024
export IMAGE_SIZE=768

# ============================================================
# Runtime settings
# ============================================================

export ACCELERATE_MIXED_PRECISION=bf16
export TORCH_DTYPE=bfloat16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ============================================================
# Sanity checks
# ============================================================

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/deepencoderv2.py" ]]; then
  echo "ERROR: missing ${MODEL_PATH}/deepencoderv2.py" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/modeling_deepseekocr2.py" ]]; then
  echo "ERROR: missing ${MODEL_PATH}/modeling_deepseekocr2.py" >&2
  exit 1
fi

if ! grep -q "scanflow_v2_residual_attn" "${MODEL_PATH}/deepencoderv2.py"; then
  echo "ERROR: MODEL_PATH/deepencoderv2.py is not the ScanFlow Pro V2 implementation." >&2
  exit 1
fi

if ! grep -q "scanflow_v2_intensity_mlp" "${MODEL_PATH}/deepencoderv2.py"; then
  echo "ERROR: MODEL_PATH/deepencoderv2.py lacks the dynamic V2 reasoning-intensity head." >&2
  exit 1
fi

if grep -q "scanflow_v2_residual_scale" "${MODEL_PATH}/deepencoderv2.py"; then
  echo "ERROR: MODEL_PATH/deepencoderv2.py still contains the obsolete static V2 lambda." >&2
  exit 1
fi

if ! grep -q "combined = torch.cat(\[visual_tokens, causal_inputs\]" "${MODEL_PATH}/deepencoderv2.py"; then
  echo "ERROR: V2 must generate H against immutable visual_tokens." >&2
  exit 1
fi

if grep -q "_update_visual_memory" "${MODEL_PATH}/deepencoderv2.py"; then
  echo "ERROR: V2 model unexpectedly contains the V4 stepwise memory update." >&2
  exit 1
fi

if ! grep -q "SCANFLOW_V2_RESET_NEW_PARAMS" "${MODEL_PATH}/modeling_deepseekocr2.py"; then
  echo "ERROR: MODEL_PATH/modeling_deepseekocr2.py lacks the ScanFlow Pro V2 reset path." >&2
  exit 1
fi

if [[ ! -f "${SWIFT_BASE_TEMPLATE}" ]]; then
  echo "ERROR: missing ms-swift template: ${SWIFT_BASE_TEMPLATE}" >&2
  exit 1
fi

if [[ ! -f "${SWIFT_DEEPSEEK_TEMPLATE}" ]]; then
  echo "ERROR: missing ms-swift template: ${SWIFT_DEEPSEEK_TEMPLATE}" >&2
  exit 1
fi

if ! grep -q "scanflow_pro_v4" "${SWIFT_DEEPSEEK_TEMPLATE}"; then
  echo "ERROR: ms-swift deepseek.py lacks the current N+T template branch." >&2
  exit 1
fi

if ! grep -q 'res\["scanpath_lengths"\] = scanpath_lengths' "${SWIFT_BASE_TEMPLATE}"; then
  echo "ERROR: ms-swift base.py lacks ScanFlow scanpath_lengths collation." >&2
  exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "ERROR: DATASET_PATH does not exist: ${DATASET_PATH}" >&2
  exit 1
fi

# Verify the required Transformers version before launching a 4-GPU job.
python - <<'PY'
import transformers
expected = "4.46.3"
actual = transformers.__version__
print(f"transformers version: {actual}")
if actual != expected:
    raise RuntimeError(
        f"This DeepSeek-OCR2/ScanFlow code expects transformers=={expected}, "
        f"but found {actual}."
    )
PY

# Fail before model loading if Slurm supplied MIG slices or fewer than four
# full H100s. DDP requires one complete model replica on every visible GPU.
python - <<'PY'
import torch

expected_devices = 4
actual_devices = torch.cuda.device_count()
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Visible CUDA devices: {actual_devices}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in this allocation.")
if actual_devices != expected_devices:
    raise RuntimeError(
        f"Expected {expected_devices} visible GPUs, found {actual_devices}."
    )

for index in range(actual_devices):
    props = torch.cuda.get_device_properties(index)
    memory_gib = props.total_memory / 1024**3
    print(f"GPU {index}: {props.name}, memory={memory_gib:.2f} GiB")
    if memory_gib < 70:
        raise RuntimeError(
            f"GPU {index} exposes only {memory_gib:.2f} GiB; "
            "this appears to be a MIG slice rather than a full H100-80GB."
        )
PY

# ============================================================
# Job summary
# ============================================================

echo "============================================================"
echo "Dynamic ScanFlow Pro V2 training from recurrent Plan-1 checkpoint"
echo "============================================================"
echo "Job ID:                         ${SLURM_JOB_ID}"
echo "Node:                           $(hostname)"
echo "Project:                        ${PROJECT_DIR}"
echo "Python:                         $(which python)"
echo "Swift:                          $(which swift)"
echo "Model:                          ${MODEL_PATH}"
echo "Dataset:                        ${DATASET_PATH}"
echo "Output:                         ${OUTPUT_DIR}"
echo "NPROC_PER_NODE:                 ${NPROC_PER_NODE}"
echo "CUDA_VISIBLE_DEVICES:           ${CUDA_VISIBLE_DEVICES:-set_by_slurm}"
echo
echo "Architecture:"
echo "  ScanFlow Pro V2:              ${SCANFLOW_PRO_V2}"
echo "  N+T template compatibility:   SCANFLOW_PRO_V4=${SCANFLOW_PRO_V4}"
echo "  dynamic causal tokens:        ${SCANFLOW_DYNAMIC_CAUSAL_TOKENS}"
echo "  legacy Plan-2 mode:           ${SCANFLOW_PLAN2}"
echo "  recurrent start state:        mean_pool(global/local visual tokens)"
echo "  reset Plan-1 parameters:      ${SCANFLOW_RESET_NEW_PARAMS}"
echo "  reset V2 residual branch:     ${SCANFLOW_V2_RESET_NEW_PARAMS}"
echo "  recurrence:                   Qwen h_t generation over immutable V"
echo "  residual timing:              once, after all H is generated"
echo "  residual map:                 R=CrossAttention(V,H,H)"
echo "  intensity gate:               lambda_i(V,Q), tokenwise and question-conditioned"
echo "  initial lambda_i:             ${SCANFLOW_V2_INITIAL_INTENSITY}"
echo "  LM visual input:              [V + lambda(V,Q)*R; H]"
echo "  text injection:               cross-attention after each Qwen layer"
echo
echo "Attention supervision:"
echo "  enabled:                      ${DEEPSEEK_OCR2_ENABLE_ATTENTION_SUPERVISION}"
echo "  attention loss weight:        ${DEEPSEEK_OCR2_ATTENTION_LOSS_WEIGHT}"
echo "  supervised Qwen layers:       ${DEEPSEEK_OCR2_ATTENTION_LAYER_INDICES}"
echo "  heads per Qwen layer:         14"
echo "  head aggregation:             mean over all 14 heads"
echo "  layer aggregation:            mean of 4 independent layer losses"
echo "  residual uses fixation map:   no"
echo
echo "Memory:"
echo "  gradient checkpointing:       enabled"
echo "  crop mode:                    ${CROP_MODE}"
echo "  DDP model replicas:           ${NPROC_PER_NODE}"
echo
echo "Trainable prefixes:"
echo "  model.qwen2_model.model"
echo "    includes scanflow_v2_visual_norm"
echo "    includes scanflow_v2_reasoning_norm"
echo "    includes scanflow_v2_residual_attn"
echo "    includes scanflow_v2_intensity_visual_norm"
echo "    includes scanflow_v2_intensity_question_norm"
echo "    includes scanflow_v2_intensity_mlp"
echo "  model.qwen2_model.text_context_proj"
echo
echo "Cache settings:"
echo "  XDG_CACHE_HOME:                ${XDG_CACHE_HOME}"
echo "  MODELSCOPE_CACHE:              ${MODELSCOPE_CACHE}"
echo "  HF_HOME:                       ${HF_HOME}"
echo "  HF_DATASETS_CACHE:             ${HF_DATASETS_CACHE}"
echo "  HF_MODULES_CACHE:              ${HF_MODULES_CACHE}"
echo "============================================================"

# ============================================================
# Training
# ============================================================

cd "${SWIFT_DIR}"

swift sft \
  --use_hf 1 \
  --model "${MODEL_PATH}" \
  --dataset "${DATASET_PATH}" \
  --remove_unused_columns false \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --fp16 false \
  --bf16 true \
  --ddp_find_unused_parameters false \
  --num_train_epochs 10 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --lr_scheduler_type constant_with_warmup \
  --warmup_ratio 0.03 \
  --weight_decay 0.02 \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner true \
  --freeze_parameters_ratio 1 \
  --trainable_parameters \
    model.qwen2_model.model \
    model.qwen2_model.text_context_proj \
  --max_grad_norm 1.0 \
  --gradient_checkpointing true \
  --max_pixels 602112 \
  --max_length 4096 \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --lazy_tokenize true \
  --dataset_shuffle true \
  --train_dataloader_shuffle true \
  --logging_steps 1 \
  --save_steps 100 \
  --save_total_limit 10 \
  --seed 42 \
  --data_seed 42 \
  --output_dir "${OUTPUT_DIR}"

echo "Job finished at: $(date)"
