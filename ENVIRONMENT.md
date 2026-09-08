Environment Guide

ScanFlow representation training and vLLM evaluation should use separate Python environments. The two workflows have different compatibility constraints, and upgrading a shared environment can break checkpoint resume or custom-model loading.

Reference platform

The draft release was developed on Digital Research Alliance of Canada H100 systems with:

StdEnv/2023
GCC
Python 3.11
CUDA 12.2
PyTorch 2.6.0+computecanada
PyArrow 21.0.0 module
OpenCV 4.12.0 module

Exact module names may differ across Fir, Rorqual, and Trillium.

Training environment

The training-side DeepSeek-OCR2 integration expects the compatibility pins in requirements-training.txt, particularly:

transformers==4.46.3
tokenizers==0.20.3
peft==0.15.2

Example local setup:

python3.11 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
export PYTHONPATH="$PWD/ms-swift:$PYTHONPATH"

On Alliance systems, packages supplied through the wheelhouse can be installed with --no-index. The +computecanada build suffix is cluster-specific and should not be placed in portable PyPI requirements.

vLLM evaluation environment

The reference evaluation environment used:

torch==2.6.0+computecanada
vllm==0.8.5.post1+computecanada
transformers==4.57.6+computecanada
tokenizers==0.22.2+computecanada
datasets==4.8.4+computecanada
accelerate==1.14.0
numpy==1.26.4+computecanada
pandas==2.3.3+computecanada
matplotlib==3.10.8+computecanada

The portable constraints are recorded in requirements-evaluation.txt without cluster-specific build suffixes.

Example setup:

python3.11 -m venv .venv-evaluation
source .venv-evaluation/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-evaluation.txt

Cluster modules

The supplied Slurm jobs use the following module pattern:

module --force purge
module load StdEnv/2023
module load gcc
module load python/3.11
module load cuda/12.2
module load arrow/21.0.0
module load opencv/4.12.0

Verify the available versions on the target cluster with module spider before submission.

Runtime paths

The draft Slurm scripts retain research-cluster path settings. Before running them in another checkout, configure or edit:

PROJECT_DIR
VENV_DIR
MODEL_PATH
DATASET_PATH or DATASETS_ROOT
OUTPUT_DIR
Slurm allocation/account

Model weights and datasets are not included in this repository.

CUDA preflight

Before a long run, verify both the Slurm allocation and Python environment:

nvidia-smi -L

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
PY

If nvidia-smi fails inside a valid GPU allocation, investigate the allocated node or driver. If nvidia-smi succeeds but PyTorch reports no CUDA device, investigate the Python/PyTorch installation.

Checkpoint resume compatibility

PEFT versions newer than the training-compatible stack may attempt to import tensor-parallel integrations absent from Transformers 4.46.3 while loading a LoRA adapter. For the documented training/resume workflow, retain peft==0.15.2 with transformers==4.46.3 unless the complete stack is deliberately migrated and revalidated.
