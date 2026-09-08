Third-Party Notices

This repository contains original research modifications together with code derived from or vendored from third-party projects. The notices below are provided for attribution and do not replace the license text shipped with each project.

CueFlow

ScanFlow was developed from the CueFlow research codebase:

Repository: https://github.com/jiaq-liu/CueFlow

Local project license: LICENSE

Files derived from CueFlow retain their applicable copyright and license terms.

DeepSeek-OCR2

The directory DeepSeek-OCR-2/ contains a modified copy of the DeepSeek-OCR2 codebase used for the ScanFlow visual encoder and vLLM runtime.

Upstream repository: https://github.com/deepseek-ai/DeepSeek-OCR-2

Retained license: DeepSeek-OCR-2/LICENSE.txt

Modified ScanFlow runtime files include the dynamic recurrent encoder, the post-recurrence cross-attention residual, the question-conditioned tokenwise intensity controller, and evaluation-analysis tensor capture.

MS-Swift

The directory ms-swift/ contains a modified source snapshot of MS-Swift used for model registration, dataset/template processing, training, and ScanFlow diagnostics.

Upstream repository: https://github.com/modelscope/ms-swift

Retained license: ms-swift/LICENSE

Upstream license: Apache License 2.0

The vendored snapshot is included because the ScanFlow integration spans multiple templates, registration files, pipeline components, and callbacks.

Runtime dependencies

The project also depends on separately installed open-source packages, including:

PyTorch;

Hugging Face Transformers;

Hugging Face PEFT;

Hugging Face Datasets;

Accelerate;

vLLM;

NumPy, pandas, Matplotlib, and scikit-learn; and

Einops, Addict, EasyDict, Pillow, and OpenCV.

These packages are not vendored by this repository unless they are explicitly present in a third-party directory. Each remains governed by its own license.

Datasets

ChartQA, ChartQAPro, ChartBench, and SalChartQA are not redistributed in this repository. Users must obtain them from their respective sources and comply with their licenses, terms, and citation requirements.

Model weights

DeepSeek-OCR2, Plan-1, ScanFlow, LoRA, optimizer, and trainer-state weights are not redistributed in this Git repository. Any future model-hosting release remains subject to the applicable upstream model license and attribution requirements.

No endorsement

References to upstream projects identify technical dependencies and sources. They do not imply endorsement of this project by the upstream authors or organizations.
