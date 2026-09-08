
# Visual Grounding and Explainability Auditing for Prompt-Driven Parameter-Efficient Radiology Report Generation

This repository contains the code and experimental resources for the paper:

**Visual Grounding and Explainability Auditing for Prompt-Driven Parameter-Efficient Radiology Report Generation**

The project investigates whether a vision-language model can generate radiology reports while also identifying the image regions that support individual findings. In addition to report-generation performance, the project evaluates the quality and reliability of visual grounding using both attribution methods and image-independent spatial baselines.

## Overview

The proposed system combines:

* **Rad-DINO** as a frozen medical vision encoder
* A lightweight **Spatial Adapter** to project visual features into the language-model embedding space
* **Qwen2.5-3B-Instruct** as the language decoder
* **QLoRA/LoRA** for parameter-efficient adaptation
* Discrete **coordinate tokens** for bounding-box generation
* Two fixed prompting modes for report generation and grounded report generation

The model generates report text and, when appropriate, bounding-box coordinates in a single autoregressive generation process.

The project also investigates an important evaluation question:

> **Does a good radiology report or a valid grounding output necessarily mean that the model is using image-specific information?**

To investigate this, generated reports and grounding outputs are evaluated separately and compared with appropriate baselines.

---

## Model Architecture

The overall pipeline is:

```text
Chest X-ray
    │
    ▼
Frozen Rad-DINO
    │
    ▼
37 × 37 visual patch features
    │
    ▼
Spatial pooling
    │
    ▼
19 × 19 = 361 visual tokens
    │
    ▼
Spatial Adapter
(768 → 2048 + LayerNorm)
    │
    ▼
Projected visual tokens
    │
    ├──────────────┐
    ▼              ▼
Prompt tokens   Report tokens
    │              │
    └──────┬───────┘
           ▼
     Qwen2.5-3B-Instruct
       + LoRA adapters
           │
           ▼
   Report + coordinate tokens
```

The vision encoder is frozen during training. Only the Spatial Adapter, LoRA parameters, and coordinate-token embeddings are trained.

---

## Two Generation Modes

The system uses two predefined prompts.

### FindGen

Generates a radiology report without localisation coordinates.

```text
Chest X-ray → Report
```

### GroundRep

Generates a report together with coordinates for groundable findings.

```text
Chest X-ray → Report + Bounding Boxes
```

The prompts are fixed and are applied automatically by the system. Users select the required mode rather than writing their own prompt.

---

## Grounding Representation

Bounding boxes are represented using discrete coordinate tokens.

Each coordinate is normalised to the range `[0, 1]` and quantised into **32 bins per axis**.

The model therefore uses:

* 32 x-coordinate values
* 32 y-coordinate values
* Object markers
* Box markers

A simplified representation is:

```text
<obj> finding text <box> <x1> <y1> <x2> <y2> </box>
```

During inference, the generated coordinate tokens are converted back into image coordinates and used to draw the predicted bounding box.

Constrained decoding is used to ensure that tokens generated inside a bounding-box span follow the required coordinate format.

---

## Datasets

The experiments use the following datasets:

### PadChest-GR

PadChest-GR is used for radiology report generation and visual grounding.

The grounding experiments use the abnormal-study subset because coordinate supervision is available for abnormal findings.

Dataset statistics used in the experiments:

| Dataset              | Train | Validation | Test | Total |
| -------------------- | ----: | ---------: | ---: | ----: |
| PadChest-GR          | 3,643 |        456 |  456 | 4,555 |
| PadChest-GR abnormal | 2,479 |        310 |  310 | 3,099 |

The complete PadChest-GR dataset is split at patient level with abnormality stratification. The abnormal-only subsets are then obtained from these partitions.

### IU X-ray

IU X-ray is used to evaluate report-generation performance.

| Dataset  | Train | Validation | Test | Total |
| -------- | ----: | ---------: | ---: | ----: |
| IU X-ray | 1,964 |        252 |  265 | 2,481 |

The splits are constructed using report hashing to reduce the possibility of reference-text leakage.

---

## Training

The model is designed for limited computational resources and was trained using a single **NVIDIA T4 GPU with 16 GB memory**.

Important training settings include:

```text
Language model: Qwen2.5-3B-Instruct
Quantisation: 4-bit NF4
Adaptation: LoRA / QLoRA
LoRA rank: 32
LoRA alpha: 64
LoRA dropout: 0.05

Image size: 518 × 518
Original visual grid: 37 × 37
Pooled visual grid: 19 × 19
Visual embedding dimension: 768
Adapter output dimension: 2048

Batch size: 1
Gradient accumulation: 8
Effective batch size: 8
Learning rate (LoRA): 2e-6
Learning rate (Adapter): 5e-5
Warm-up: 3%
Weight decay: 0.01
Gradient clipping: 1.0
Epochs: 3
Random seed: 42
```

The vision encoder remains frozen during training.

---

## Parameter Efficiency

The proposed system adapts only a small fraction of the complete model.

Approximately:

```text
Total parameters:       3.2B
Trainable parameters:   65.7M
Trainable fraction:     ~2.1%
```

The trainable parameters mainly consist of:

* LoRA adapters
* Spatial Adapter
* Coordinate-token embeddings

This allows the model to be trained using substantially fewer resources than full model fine-tuning.

---

## Evaluation

The project evaluates the model from several perspectives rather than relying on report-generation scores alone.

### 1. Report Generation

The following natural language generation metrics are used:

* BLEU-1
* BLEU-2
* BLEU-3
* BLEU-4
* ROUGE-L
* METEOR
* BERTScore

Report diversity is also evaluated using:

* Uniqueness
* Most-common report frequency
* Distinct-1
* Mean generated report length

A **constant-report baseline** is included to test whether reasonable report scores can be achieved without using the input image.

---

### 2. Visual Grounding

Grounding performance is evaluated using:

* Mean IoU (mIoU)
* IoU > 0.1
* IoU > 0.25
* Mass-in-Box
* Grounding coverage
* Invalid-box rate

The model is compared with:

#### Decoder-based methods

* Free sampled decoding
* Constrained decoding

#### Post-hoc attribution methods

* Grad-CAM
* HiRes-CAM
* Score-CAM

#### Image-independent spatial baselines

* Uniform prior
* Centre prior
* Lung prior
* Edge prior

These baselines are important because a localisation result should not automatically be interpreted as evidence of image-specific reasoning.

---

### 3. Cross-Dataset Grounding Evaluation

The grounding approach is also evaluated on **NIH ChestX-ray8** using the Pointing Game.

This provides an additional test of whether the generated localisation corresponds to clinically relevant image regions outside the main training dataset.

---

## Explainability Analysis

The repository also includes code for analysing post-hoc visual explanations.

The project compares CAM-based methods with simple spatial priors because a visually convincing saliency map does not necessarily indicate clinically correct localisation.

For example, the experiments investigate whether attribution methods perform better than a simple fixed centre region.

This helps distinguish:

```text
Model-generated explanation
        ≠
Clinically correct localisation
```

---

## Representation Diagnostics

The visual interface is also analysed using representation-level diagnostics.

These include:

* Mean pairwise cosine similarity
* Signal-to-shared ratio
* Visual-token norm comparison
* Report diversity

These diagnostics were used to identify representation collapse and to compare different visual-to-language interface designs.

---



## Installation

Create a Python environment and install the required packages:

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

For GPU training, make sure that the installed PyTorch, CUDA, and `bitsandbytes` versions are compatible with the local environment.

---

## Data Preparation

The datasets are **not included in this repository**.

Please obtain the datasets from their official sources and follow their respective terms of use and research licences.

After downloading the data, update the dataset paths in the configuration file.

For example:

```python
DATA_ROOT = "/path/to/dataset"
```

The repository does not redistribute the original medical images or patient data.

---

## Running Training

After preparing the datasets and configuration:

```bash
python training/train.py
```

The random seed should be set to `42` to reproduce the experimental configuration.

---

## Running Inference

Report generation:

```bash
python inference/generate.py --mode findgen
```

Grounded report generation:

```bash
python inference/generate.py --mode groundrep
```

The grounded mode generates report text together with coordinate tokens, which are subsequently decoded into bounding boxes.

---

## Running Evaluation

Report-generation evaluation:

```bash
python evaluation/report_metrics.py
```

Grounding evaluation:

```bash
python evaluation/grounding_metrics.py
```

CAM-based evaluation:

```bash
python evaluation/cam_methods.py
```

The evaluation scripts assume that the required predictions, reference annotations, and dataset files have been prepared.

---

## Automated Tests

The repository contains tests for deterministic components of the pipeline.

These include:

* Coordinate quantisation
* Coordinate decoding
* Bounding-box normalisation
* IoU calculation
* Grounded-report formatting
* Report-only formatting
* Groundability checks
* Study-key generation
* Generation parsing
* Dataset split integrity

Run the tests using:

```bash
pytest
```

The tests mainly verify deterministic pipeline behaviour and data integrity. They do not establish clinical correctness of the generated reports or bounding boxes.

---

## Reproducibility

The main experimental configuration uses:

```text
Random seed: 42
GPU: NVIDIA T4 16 GB
Mixed precision: FP16
Quantisation: 4-bit NF4
```

Dataset partitions and model configuration should be kept fixed when reproducing the reported results.

Because generation involves stochastic decoding for some experiments, sampled results may vary slightly between runs.

---

## Limitations

The released implementation has several limitations.

1. **Report under-generation:** generated reports are shorter than the reference reports.
2. **Limited grounding accuracy:** the generated bounding boxes achieve relatively low spatial overlap.
3. **Single-finding generation:** the current decoding strategy limits stable generation to one grounded finding.
4. **Abnormal-only grounding training:** the grounding model is trained on abnormal studies and therefore does not learn study-level abstention for completely normal examinations.
5. **Limited computational resources:** the vision encoder is frozen and visual tokens are spatially pooled to reduce memory requirements.
6. **Attribution comparison:** CAM methods and decoder-generated boxes use different mechanisms and are therefore not perfectly equivalent comparisons.

These limitations are discussed in detail in the accompanying paper.

---

## Citation

If you use this code or build on this work, please cite the associated paper:

```bibtex
@inproceedings{YOUR_CITATION_KEY,
  title     = {Visual Grounding and Explainability Auditing for Prompt-Driven Parameter-Efficient Radiology Report Generation},
  author    = {Nafisa Islam Rifa},
  booktitle = {MICAD 2026},
  year      = {2026}
}
```

Please replace the BibTeX entry above with the final citation provided by the conference or publisher.

---

## Acknowledgements

This work was completed as part of an MSc Data Science project at the **University of Greenwich**.

The project uses publicly available medical imaging datasets and pretrained models. Please refer to the original dataset and model publications when using the corresponding resources.

---

## Licence

The code in this repository is released under the licence specified in `LICENSE`.

The datasets, pretrained models, and other third-party resources remain subject to their original licences and terms of use.
