import os
import logging
import torch
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
    EvalPrediction
)
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from torch import Tensor
from typing import List, Dict, Any

# 1. Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Definicja etykiet
entity_types = ["Chemical", "Disease"]
label_list = ['O'] + [f'B-{et}' for et in entity_types] + [f'I-{et}' for et in entity_types]


# 3. Funkcja spłaszczania
def flatten_data_kb(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    new_texts = []
    new_entities = []
    for i in range(len(batch.get("document_id", batch.get("pmid", [])))):
        passage_list = batch["passages"][i]
        try:
            passage_texts = [p["text"][0] for p in passage_list if p.get("text")]
        except IndexError:
            passage_texts = []
        full_text = " ".join(passage_texts)
        all_entities = batch["entities"][i]
        new_texts.append(full_text)
        new_entities.append(all_entities)
    return {"text": new_texts, "entities": new_entities}


# 4. Funkcja przetwarzania
def convert_to_tokens_and_labels(
        samples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerFast,
        label_list_map: List[str]
) -> Dict[str, List[Any]]:
    label2id_map = {l: i for i, l in enumerate(label_list_map)}
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for sample in samples:
        text = sample["text"]
        entities = sample["entities"]
        char_labels = ['O'] * len(text)
        for entity_dict in entities:
            entity_type = entity_dict['type']
            if entity_type not in entity_types:
                continue
            for (start, end) in entity_dict['offsets']:
                if start < len(text) and end <= len(text):
                    char_labels[start] = f"B-{entity_type}"
                    for i in range(start + 1, end):
                        if i < len(text):
                            char_labels[i] = f"I-{entity_type}"

        tokenized = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
        offsets = tokenized['offset_mapping']
        labels = []
        for (start, end) in offsets:
            if start == end:
                labels.append(-100)
            else:
                char_label_str = char_labels[start]
                labels.append(label2id_map.get(char_label_str, label2id_map['O']))

        all_input_ids.append(tokenized['input_ids'])
        all_attention_masks.append(tokenized['attention_mask'])
        all_labels.append(labels)

    return {"input_ids": all_input_ids, "attention_mask": all_attention_masks, "labels": all_labels}


# 5. Dataset
class NERDataset(TorchDataset):
    def __init__(self, encodings: Dict[str, List[Any]]):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


# 6. Funkcja treningu (TYLKO PROBE)
def run_linear_probe_benchmark(
        model_name: str,
        train_data: List[Dict[str, Any]],
        dev_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        label_list: List[str],
        output_dir: str,
        epochs: int
) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    id2label_map = {i: l for i, l in enumerate(label_list)}
    label2id_map = {l: i for i, l in enumerate(label_list)}

    logger.info("Tokenizacja danych...")
    train_encodings = convert_to_tokens_and_labels(train_data, tokenizer, label_list)
    dev_encodings = convert_to_tokens_and_labels(dev_data, tokenizer, label_list)
    test_encodings = convert_to_tokens_and_labels(test_data, tokenizer, label_list)

    train_dataset = NERDataset(train_encodings)
    dev_dataset = NERDataset(dev_encodings)
    test_dataset = NERDataset(test_encodings)

    logger.info(f"Ładowanie modelu do benchmarku: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label_map,
        label2id=label2id_map
    )

    # --- ZAMRAŻANIE (FROZEN BODY) ---
    logger.info("ZAMRAŻANIE WAG (Linear Probe Mode). Trenowana tylko głowa.")
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Odmrażanie głowy klasyfikującej
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, "qa_outputs"):
        for param in model.qa_outputs.parameters():
            param.requires_grad = True
    # --------------------------------

    metric = evaluate.load("seqeval")

    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        preds = np.argmax(p.predictions, axis=2)
        labels = p.label_ids
        true_preds = [
            [label_list[p] for (p, l) in zip(prediction, label_row) if l != -100]
            for prediction, label_row in zip(preds, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label_row) if l != -100]
            for prediction, label_row in zip(preds, labels)
        ]
        results = metric.compute(predictions=true_preds, references=true_labels, zero_division=0)

        # ZWRACAMY WSZYSTKIE METRYKI
        return {
            "overall_f1": results["overall_f1"],
            "overall_precision": results["overall_precision"],
            "overall_recall": results["overall_recall"],
            "overall_accuracy": results["overall_accuracy"]
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        learning_rate=1e-3,  # Wyższy LR dla samej głowy
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
        compute_metrics=compute_metrics
    )

    logger.info("Rozpoczynanie kalibracji głowy (Linear Probe Training)...")
    trainer.train()

    logger.info("Ewaluacja końcowa na zbiorze testowym...")
    metrics = trainer.evaluate(test_dataset)
    return metrics


def main() -> None:
    # 1. Setup
    models = {
        "BERT-base": "bert-base-uncased",
        "BioBERT": "dmis-lab/biobert-base-cased-v1.1",  # jest "uncased" mimo nazwy
        "BioClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
        "PubMedBERT": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    }

    dataset = load_dataset("bigbio/bc5cdr", name="bc5cdr_bigbio_kb", trust_remote_code=True)
    flattened_dataset = dataset.map(flatten_data_kb, batched=True, num_proc=os.cpu_count() or 1)

    train_data = list(flattened_dataset["train"])
    dev_data = list(flattened_dataset["validation"])
    test_data = list(flattened_dataset["test"])

    print("\n" + "=" * 80)
    print("START BENCHMARKU (LINEAR PROBE / FROZEN BODY)")
    print("Ten tryb sprawdza wiedzę zawartą w modelu BEZ pełnego douczania.")
    print("=" * 80)

    for name, model_name in models.items():
        print(f"\nModel: {name}")

        # Uruchamiamy tylko wersję "Linear Probe"
        results = run_linear_probe_benchmark(
            model_name=model_name,
            train_data=train_data, dev_data=dev_data, test_data=test_data,
            label_list=label_list,
            output_dir=f"./results/{name}_benchmark_probe",
            epochs=8
        )

        print("-" * 80)
        print(f"WYNIKI BENCHMARKU DLA {name}:")
        print(f"  F1 Score:   {results['eval_overall_f1']:.4f}")
        print(f"  Precyzja:   {results['eval_overall_precision']:.4f}")
        print(f"  Czułość:    {results['eval_overall_recall']:.4f}")
        print(f"  Dokładność: {results['eval_overall_accuracy']:.4f}")
        print("-" * 80)


if __name__ == "__main__":
    main()