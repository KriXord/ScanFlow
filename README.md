# ScanFlow

ScanFlow is a research framework for integrating human scanpaths into the latent visual reasoning process of multimodal large language models. Its central idea is that human visual attention is not merely a static saliency distribution: it is a temporally ordered process in which successive fixations reflect an evolving reasoning trajectory.

ScanFlow therefore learns question-conditioned latent states that progress through a scanpath over time. Each state represents accumulated visual reasoning and preserves information from earlier steps, while supervision encourages the model's internal visual attention to follow the spatial and temporal structure of human scanpaths.

This repository is currently a draft research release. Model weights and datasets are not included.

## Motivation

Most saliency-guided vision-language systems represent human attention as a single spatial map. This can identify visually important regions, but it removes the temporal ordering that distinguishes a scanpath from static saliency.

For visual question answering, order matters. A person may first locate the chart title, then identify the relevant axis or legend, and finally inspect the marks needed to answer the question. Collapsing these fixations into one map preserves *where* people looked but discards *when* and *in what reasoning sequence* they looked there.

ScanFlow investigates whether this ordered behavior can be internalized as latent visual reasoning. Its goal is to make temporal order correspond to actual computation: later reasoning states should be generated from the visual evidence and reasoning history accumulated by earlier states, rather than merely occupying later positions in a parallel sequence.

## Core challenges

### Temporal latent collapse

An early failure mode was temporal collapse. Successive latent states could produce nearly identical attention distributions:

```text
A_1 ≈ A_2 ≈ ... ≈ A_T
```

The resulting maps could still appear spatially plausible, but they represented repeated saliency rather than a changing scanpath. The model learned broadly *where* useful information was located without learning *when* different regions should be examined.

### Shortcut learning

ScanFlow also exposed architectural shortcuts. Learnable initialization, auxiliary fixation decoders, or jointly adapting modules could reduce the supervision loss without requiring the recurrent hidden states to carry meaningful temporal information.

In that case, the model could predict average or question-type-specific saliency patterns without using the intended temporal reasoning path. This motivated stricter separation between the recurrent reasoning process and the components used to supervise or evaluate it.

## Architecture

Let:

- $V = \{v_i\}_{i=1}^{N}$ denote the visual-token sequence;
- $Q$ denote the question-token representation;
- $H = \{h_t\}_{t=1}^{T}$ denote the latent reasoning trajectory; and
- $S_t$ denote the human fixation target at scanpath step $t$.

ScanFlow generates reasoning states recurrently:

```text
h_t = G(V, Q, h_1, ..., h_(t-1))
```

Each new state is conditioned on the original multimodal context and the preceding reasoning history. This makes the temporal index an explicit computational dependency rather than only a positional label.

The model's internal visual attention at each step is then compared with the corresponding ordered human fixation target:

```text
A_t = VisualAttention(h_t, V)
L_scanpath = sum_t AlignmentLoss(A_t, S_t)
```

The fixation maps are supervision and interpretation targets for the latent trajectory. The hidden states themselves are not intended to become isolated fixation embeddings; each state retains accumulated context and reasoning history.

## Design progression

| Stage | Temporal representation | Main limitation or contribution |
| --- | --- | --- |
| Static saliency | One spatial distribution | Preserves important locations but discards fixation order |
| Plan-1 Parallel | Ordered latent positions generated together | Represents order, but does not make later states depend on earlier computation |
| Recurrent ScanFlow | Stepwise latent generation with causal history | Makes temporal order part of the reasoning computation |
| Direct attention supervision | Internal visual attention aligned with ordered fixations | Reduces reliance on a separate learnable fixation decoder |
| Frozen diagnostic components and ablations | Supervision and evaluation isolated from the reasoning path | Exposes shortcut learning and tests whether temporal information is genuinely carried by the latent states |

The historical Plan-1 Parallel implementation is retained under `baselines/plan1_parallel/` to document the distinction between an ordered representation and an ordered computation. It is not the active ScanFlow architecture.

## Repository layout

```text
.
├── modelings/scanflow/             # Hugging Face and training-side ScanFlow model code
├── baselines/plan1_parallel/       # Earlier parallel baseline
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
├── .env.example
├── ENVIRONMENT.md
└── THIRD_PARTY_NOTICES.md
```

Some filenames and environment variables retain historical development names such as `scanflow_pro_*` and `SCANFLOW_V2_*`. They are kept for checkpoint and runtime compatibility and do not change the scope of this repository: this release documents ScanFlow.

## Installation

Training and vLLM evaluation use different dependency stacks. Create separate environments instead of installing both requirement files into the same environment.

### Training environment

```bash
python -m venv .venv-training
source .venv-training/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
export PYTHONPATH="$PWD/ms-swift:${PYTHONPATH:-}"
```

### Evaluation environment

```bash
python -m venv .venv-evaluation
source .venv-evaluation/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-evaluation.txt
```

Alliance/Compute Canada users should follow the module and wheelhouse guidance in [`ENVIRONMENT.md`](ENVIRONMENT.md).

## Data

Datasets are intentionally excluded from version control. The current evaluation workflow supports:

- ChartQA
- ChartQAPro
- SalChartQA

SalChartQA provides the ordered fixation annotations used to supervise and analyze scanpath behavior. ChartQA and ChartQAPro are used to evaluate downstream visual question answering and generalization.

Place datasets under `datasets/`, or provide the dataset root through the applicable command-line option or environment variable. Users are responsible for obtaining each dataset under its original license and terms.

## Training

The checked-in training job records the research configuration used to train the released ScanFlow representation. Before submitting `train_scanflowpro_v2.slurm`, configure the cluster account, environment, initialization checkpoint, dataset, and output paths for the target system.

The implementation preserves reset boundaries between previously trained recurrent components and newly introduced experiment-specific modules. Consult the comments and preflight checks in the Slurm script before starting or resuming a run.

## Evaluation and analysis

ScanFlow evaluation considers both downstream task performance and whether the latent trajectory exhibits meaningful temporal visual behavior.

### Downstream question answering

The evaluation workflow produces per-sample predictions for ChartQA, ChartQAPro, and SalChartQA. Benchmarks can be evaluated independently by passing one dataset after `--datasets`.

### Temporal scanpath behavior

The central behavioral analysis examines:

- stepwise visual-attention distributions;
- alignment with ordered human fixation targets;
- changes in attended regions across reasoning steps;
- temporal diversity versus repeated-map collapse; and
- differences in scanpath behavior across questions and benchmarks.

These analyses distinguish a genuinely evolving latent trajectory from a model that repeatedly predicts a static saliency pattern.

Example single-dataset evaluation:

```bash
python run_scanflowpro_v2_evaluation.py \
  --datasets SalChartQA \
  --model-name /path/to/scanflow-model \
  --datasets-root /path/to/datasets \
  --output-dir eval_outputs/salchartqa
```

Some evaluation entry points retain historical filenames for compatibility. Their names should not be interpreted as defining a separate project scope.

## Model weights

Model checkpoints, optimizer state, and trainer state are excluded from Git because they are multi-gigabyte artifacts. A Hugging Face model link will be added when the inference checkpoint is published.

## Acknowledgements and licensing

ScanFlow builds on CueFlow, DeepSeek-OCR2, MS-Swift, Hugging Face Transformers and PEFT, and vLLM. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the license files retained in the vendored directories.

## Status

This repository currently represents a draft ScanFlow code release. Paths, installation automation, model hosting, and final experimental results may be refined before an archival release.
