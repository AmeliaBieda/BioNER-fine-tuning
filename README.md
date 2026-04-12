# Named Entity Recognition and Relation Extraction

This repository contains the practical implementation of my engineering thesis. The project explores the application of Transformer-based language models in advanced Biomedical Natural Language Processing (BioNLP) tasks. It implements a pipeline architecture covering Named Entity Recognition (NER) and Relation Extraction (RE) using the medical document corpus BC5CDR.

## Key Features & Engineering Problem Solving

* **Domain Adaptation Analysis:** Conducted a comprehensive comparative study between a general-purpose model (BERT-Base) and highly specialized models (BioBERT, BioClinicalBERT, PubMedBERT) to evaluate the impact of pre-training domains.
*  **NLP Pipeline Architecture:** * Stage 1 (NER): Fine-tuned models for sequence labeling to identify diseases and chemicals using the IOB tagging scheme.
    *  Stage 2 (RE): Extracted causal semantic relations (e.g., Chemical-Induced Disease) between identified entities by leveraging entity markers.
* **Handling Highly Imbalanced Data:** Overcame the challenge of dominant negative classes in the RE stage by implementing a custom Focal Loss function and undersampling techniques, significantly stabilizing the training process.

## Key Results

* **Named Entity Recognition:** PubMedBERT achieved the highest performance (F1 = 88.62%). This proved that a domain-specific WordPiece vocabulary prevents excessive token fragmentation, drastically improving precision.
* **Relation Extraction:** BioBERT achieved the best results (F1 = 69.51%), successfully understanding complex sentence syntax for entity pair classification.

## Tech Stack

* Language: Python 3.10.
* [Deep Learning: PyTorch (tensor operations, custom loss functions).
* NLP Ecosystem: Hugging Face `transformers` & `datasets`.
* Evaluation & ML: Scikit-learn, `seqeval`.

## Repository Structure

* **`benchmark.py`:** Script for zero-shot / base knowledge evaluation (linear probe strategy with frozen encoder weights).
* **`BC5CDR_fine-tuning.py`:** Module responsible for full model fine-tuning for the NER task.
* **`RE_fine-tuning.py`:** Pipeline module for Relation Extraction, featuring sequence classification and custom training logic (`FocalLossTrainer`).

## How to Run

The project is structured as a multi-stage experimental pipeline. To reproduce the results, please execute the modules in the following order:

1. Clone the repository: `git clone https://github.com/AmeliaBieda/BioNER-fine-tuning`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Baseline Benchmark: Run `python benchmark.py` to evaluate the initial biomedical knowledge of the models using the "frozen weights" (linear probe) strategy.   
4. NER Fine-tuning: Run `python BC5CDR_fine-tuning.py`. This script performs full fine-tuning for named entity recognition and saves the best-performing models.
5. Relation Extraction (RE): Run `python RE_fine-tuning.py`. This final module loads the weights saved during the NER stage to perform Chemical-Induced Disease (CID) relation classification.
