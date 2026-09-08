ScanFlow

ScanFlow is a research framework for reasoning-guided visual representation learning in multimodal language models. It augments static visual tokens with recurrent latent reasoning states and uses those states to modify the visual representation before it is passed to the downstream language model.

This repository is a draft research release. Model weights and datasets are not included.

Motivation

Providing an MLLM with both unchanged visual tokens and reasoning tokens can create a latent-bypass problem: the language model may answer directly from the original visual representation and ignore the reasoning trajectory. Removing the original visual tokens avoids that bypass, but can discard global context.

ScanFlow addresses this trade-off by preserving the original visual representation while applying a question-conditioned reasoning update to it.

Architecture

Let

V = {v_i}_{i=1}^N denote the original visual tokens;

Q denote the question-token representation; and

H = {h_t}_{t=1}^T denote the recurrent latent reasoning trajectory.

The reasoning trajectory is first generated against an immutable copy of V. After all recurrent states have been produced, ScanFlow computes a one-shot reasoning residual:

R = CrossAttention(LN(V), LN(H), LN(H)).

A tokenwise, question-conditioned intensity controller determines how strongly each visual location is updated:

lambda_i(V, Q) = sigmoid(F_I([LN(v_i), pool(LN(V)), pool(LN(Q))])).

The final visual representation and downstream multimodal sequence are

V'_i = V_i + lambda_i(V, Q) R_i
X_LLM = [V'; H].

This gives ScanFlow two forms of adaptivity:

spatial adaptivity: each visual token receives its own residual and intensity value;

reasoning-intensity adaptivity: the update strength depends on the image and question.

The visual update is computed only after the full reasoning sequence is generated. It therefore does not alter the recurrent trajectory H and does not introduce a temporal visual-memory update.

Architecture progression

Variant

Downstream representation

Role in this repository

Plan-1 Parallel

[V; H]

Historical baseline preserved under baselines/plan1_parallel/

ScanFlow

[V + lambda_i(V,Q)R; H]

Final architecture in this repository

ScanFlow Pro

[M^T; H] with stepwise perception updates

Separate temporal architecture; not included here

Plan-1 Parallel is retained to make the architectural progression reproducible. It is not the active ScanFlow runtime.

Repository layout

.
├── modelings/scanflow/             # Hugging Face / training-side ScanFlow model code
├── baselines/plan1_parallel/       # Earlier Plan-1 Parallel model and vLLM code
├── DeepSeek-OCR-2/                 # Vendored and modified DeepSeek-OCR2 runtime
├── ms-swift/                       # Vendored and modified MS-Swift training framework
├── dataloader.py                   # Evaluation dataset loading
├── train_dataloader.py             # Training dataset preparation
├── prompt_templates.py             # Benchmark-specific prompts
├── train_scanflowpro_v2.slurm      # Representation-training job
├── run_scanflowpro_v2_evaluation.py
├── run_scanflowpro_v2_evaluation.slurm
├── requirements-training.txt
├── requirements-evaluation.txt
├── ENVIRONMENT.md
└── THIRD_PARTY_NOTICES.md

The filenames scanflow_pro_deepencoder.py, scanflow_pro_deepseek_ocr2.py, and several SCANFLOW_PRO_* environment variables are retained as historical compatibility identifiers. In this repository those files implement the final ScanFlow architecture above. The temporal ScanFlow Pro architecture is not present.

SCANFLOW_PRO_V4=1 is also retained in some scripts because the patched MS-Swift template uses that legacy flag to reserve N + T image-token positions. It does not enable a V4 visual-memory update.

Installation

Training and vLLM evaluation use different dependency stacks. Create separate environments rather than installing both requirement files into one environment.

Training environment

python -m venv .venv-training
source .venv-training/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
export PYTHONPATH="$PWD/ms-swift:$PYTHONPATH"

Evaluation environment

python -m venv .venv-evaluation
source .venv-evaluation/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-evaluation.txt

Alliance/Compute Canada users should follow the module and wheelhouse instructions in ENVIRONMENT.md.

Data

Datasets are intentionally excluded from version control. The current loaders support:

ChartQA;

ChartQAPro;

ChartBench; and

SalChartQA, including ordered scanpath supervision.

Place datasets under datasets/ or provide the dataset root through the applicable command-line option or environment variable. Users are responsible for obtaining each dataset under its original terms.

Training

ScanFlow representation training starts from the trained recurrent Plan-1 representation and initializes only the new residual cross-attention and dynamic-intensity modules. The intended reset policy is:

export SCANFLOW_RESET_NEW_PARAMS=0
export SCANFLOW_V2_RESET_NEW_PARAMS=1
export SCANFLOW_V2_INITIAL_INTENSITY=0.001

The recurrent reasoning path is preserved, while the new intensity head initially produces lambda_i = 0.001.

Before submitting train_scanflowpro_v2.slurm, configure the account, environment, input model, dataset, and output paths for the target cluster. The checked-in Slurm file records the research configuration used for the draft release.

Evaluation and analysis

run_scanflowpro_v2_evaluation.py supports benchmark evaluation and two architecture-focused analyses.

Dynamic reasoning-intensity distribution

The evaluator records the tokenwise lambda_i(V,Q) distribution for each sample and aggregates its statistics by benchmark. This is intended to test whether harder question sets elicit stronger reasoning-conditioned perception updates.

Reasoning-modified perception

The analysis captures:

original visual tokens V;

reasoning residual R;

intensity map lambda_i(V,Q);

effective update lambda_i R_i;

modified visual tokens V'; and

projected original and modified tokens received by the language model.

Supported visualizations include update-magnitude heatmaps, relative-L2 change, cosine-direction change, shared-PCA pseudo-images, and overlays on the original chart.

Example single-dataset evaluation:

python run_scanflowpro_v2_evaluation.py \
  --datasets ChartQA \
  --model-name /path/to/scanflow-model \
  --datasets-root /path/to/datasets \
  --output-dir eval_outputs/chartqa \
  --v2-analysis

To evaluate another benchmark independently, keep only that benchmark after --datasets.

Model weights

Model checkpoints, optimizer state, and trainer state are excluded because they are multi-gigabyte artifacts and are not appropriate for ordinary Git storage. A model-hosting or archival link can be added here when weights are released.

Acknowledgements and licensing

ScanFlow builds on CueFlow, DeepSeek-OCR2, MS-Swift, Hugging Face Transformers/PEFT, and vLLM. See THIRD_PARTY_NOTICES.md and the license files retained in the vendored directories.

Status

This repository currently represents a draft code release. Paths, installation automation, model hosting, and final experimental results may be refined before an archival release.
