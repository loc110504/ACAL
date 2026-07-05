# ACAL

<p align="center">
  <a href="https://arxiv.org/abs/2602.18916">
    <img src="https://img.shields.io/badge/Preprint-arXiv-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv preprint">
  </a>
  <img src="https://img.shields.io/badge/Conference-Canadian%20AI%202026-1f6feb?style=for-the-badge" alt="Canadian AI 2026">
  <img src="https://img.shields.io/badge/Implementation-Official-2ea44f?style=for-the-badge" alt="Official implementation">
</p>

## About

- This repository contains the official implementation accompanying our paper: **Neuro-Symbolic Adaptive Collaboration of Arena-Based Argumentative LLMs for Contestable Legal Reasoning**

- ACAL integrates adaptive multi-agent collaboration with an Arena-based Quantitative Bipolar Argumentation Framework (A-QBAF). The framework constructs structured arguments, resolves conflicting claims, computes argument strengths, and supports human intervention in the reasoning process.


## Framework

<p align="center">
  <img src="framework.png" alt="Overview of the ACAL framework" width="90%">
</p>


## Installation

### Prerequisites

* Python <= 3.11
* Conda or another Python environment manager
* Access credentials for the required model APIs

### Clone the Repository

```bash
git clone https://github.com/loc110504/ACAL.git
cd ACAL
```

### Create the Environment

We recommend using Conda to create an isolated environment:

```bash
conda create -n acal python=3.11 -y
conda activate acal
pip install -r requirements.txt
```

## API Configuration

Create a `.env` file in the repository root and provide the required credentials:

```dotenv
# Google Gemini API
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

# Azure OpenAI credentials used by the RAG components
AZURE_API_KEY="YOUR_AZURE_API_KEY"
AZURE_ENDPOINT="YOUR_AZURE_ENDPOINT"
```

## Running

Run the main experimental pipeline from the repository root:

```bash
python main.py
```

Before running an experiment, verify that:

* all dependencies are installed;
* the required API credentials are available in `.env`;
* the input data and task configuration are correctly selected; and
* the configured model deployments are accessible.

---

## Reproducibility Notes

ACAL relies on API-based large language models. Experimental outputs may vary across model versions, provider-side updates, deployment configurations, and sampling settings.

For reproducible experiments, record and report:

* the exact model and deployment identifiers;
* API or model versions when available;
* prompts and agent-role definitions;
* decoding and sampling parameters;
* retrieval configuration;
* ACAL hyperparameters, including `base_adjustment`; and
* dataset splits and evaluation settings.


## Citation

```bibtex
@inproceedings{cao2026neuro,
  title={Neuro-Symbolic Adaptive Collaboration of Arena-Based Argumentative LLMs for Contestable Legal Reasoning},
  author={Cao, Hoang-Loc and Ho, Phuc and Nguyen, Truong Thanh Hung and Nguyen, Phuc Truong Loc and Nguyen, Dinh Thien Loc and Cao, Hung},
  booktitle={The 39th Canadian Conference on Artificial Intelligence},
  pages={895--902},
  year={2026},
  organization={PMLR}
}
```


