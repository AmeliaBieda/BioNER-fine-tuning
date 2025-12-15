import os
import logging
import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Dict, Any, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
    EvalPrediction,
    logging as tr_logging
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
tr_logging.set_verbosity_error()

# 1. KONFIGURACJA
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Specjalne tokeny entity markers
SPECIAL_TOKENS = {
    "additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]
}


# 2. KLASA DATASET
class RelationDataset(TorchDataset):
    def __init__(self, encodings: Dict[str, Any], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


# 3. CUSTOM TRAINER (FOCAL LOSS)
class FocalLossTrainer(Trainer):
    """
    Trainer z zaimplementowanym Focal Loss dla niezbalansowanych danych.
    """

    def __init__(self, gamma=2.0, alpha=0.25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        return (focal_loss, outputs) if return_outputs else focal_loss


# 4. PRZETWARZANIE DANYCH
def prepare_relation_data(
        batch_data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerFast,
        negative_ratio: float = 0.1
) -> Tuple[Dict[str, List], List[int]]:
    processed_texts = []
    labels = []
    pos_count = 0
    neg_count = 0

    for doc in batch_data:
        passages = doc.get("passages", [])
        full_text = " ".join([p["text"][0] for p in passages if p.get("text")])
        entities = doc.get("entities", [])
        relations = doc.get("relations", [])

        # Zbiór relacji Gold Standard
        gold_relations = set()
        for rel in relations:
            gold_relations.add(tuple(sorted((rel["arg1_id"], rel["arg2_id"]))))

        chemicals = [e for e in entities if e["type"] == "Chemical"]
        diseases = [e for e in entities if e["type"] == "Disease"]

        for chem in chemicals:
            for dis in diseases:
                pair_key = tuple(sorted((chem["id"], dis["id"])))
                label = 1 if pair_key in gold_relations else 0

                # Undersampling negatywnych przykładów
                if label == 0 and random.random() > negative_ratio:
                    continue

                if label == 1:
                    pos_count += 1
                else:
                    neg_count += 1

                # Wstawianie markerów [E1]...[/E1] i [E2]...[/E2]
                c_start, c_end = chem["offsets"][0]
                d_start, d_end = dis["offsets"][0]

                spans = [(c_start, c_end, "[E1]", "[/E1]"), (d_start, d_end, "[E2]", "[/E2]")]
                spans.sort(key=lambda x: x[0], reverse=True)

                marked_text = full_text
                for start, end, tag_start, tag_end in spans:
                    if start < len(marked_text) and end <= len(marked_text):
                        marked_text = (
                                marked_text[:start] +
                                tag_start + " " + marked_text[start:end] + " " + tag_end +
                                marked_text[end:]
                        )

                processed_texts.append(marked_text)
                labels.append(label)

    logger.info(f"Generowanie par zakończone. Pozytywnych: {pos_count}, Negatywnych: {neg_count}")

    tokenized = tokenizer(
        processed_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors=None
    )

    return tokenized, labels


# 5. METRYKI
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = (preds == labels).mean()
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {
        'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


# 6. ANALIZA STATYSTYK ZBIORU
def analyze_and_save_stats(train_data, test_data, output_file):
    """
    Zlicza relacje i zapisuje statystyki do pliku.
    """
    def get_stats(data, name):
        total_rels = 0
        rel_types = {}

        for doc in data:
            rels = doc.get("relations", [])
            total_rels += len(rels)
            for r in rels:
                r_type = r.get("type", "Unknown")
                rel_types[r_type] = rel_types.get(r_type, 0) + 1

        return total_rels, rel_types

    train_total, train_types = get_stats(train_data, "Train")
    test_total, test_types = get_stats(test_data, "Test")

    report = []
    report.append("=== STATYSTYKI ZBIORU DANYCH BC5CDR ===")
    report.append(f"\nZBIÓR TRENINGOWY:")
    report.append(f"Liczba dokumentów: {len(train_data)}")
    report.append(f"Całkowita liczba relacji: {train_total}")
    report.append(f"Typy relacji: {train_types}")
    report.append(f"\nZBIÓR TESTOWY:")
    report.append(f"Liczba dokumentów: {len(test_data)}")
    report.append(f"Całkowita liczba relacji: {test_total}")
    report.append(f"Typy relacji: {test_types}")
    report.append("\n=======================================")
    report_text = "\n".join(report)
    print(report_text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nStatystyki zapisano w pliku: {output_file}")


# 7. FUNKCJA TRENINGOWA
def train_re_model(
        model_path: str,  # Ścieżka do modelu bazowego
        train_data: List[Dict],
        test_data: List[Dict],
        output_dir: str,
        resume_from: str = None
):
    # 1. Ładowanie tokenizera i dodanie tokenów
    # Jeśli wznawiamy, tokenizer też może być w folderze checkpointu, ale base path jest bezpieczniejszy
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_added_toks = tokenizer.add_special_tokens(SPECIAL_TOKENS)

    # 2. Przygotowanie danych
    logger.info("Tokenizacja danych treningowych...")
    train_encodings, train_labels = prepare_relation_data(train_data, tokenizer, negative_ratio=0.1)

    logger.info("Tokenizacja danych testowych...")
    test_encodings, test_labels = prepare_relation_data(test_data, tokenizer,
                                                        negative_ratio=1.0)  # 1.0 = bierzemy wszystko do testów

    train_dataset = RelationDataset(train_encodings, train_labels)
    test_dataset = RelationDataset(test_encodings, test_labels)

    # 3. Ładowanie modelu
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))

    # 4. Argumenty treningu
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,

        warmup_steps=500,
        max_steps=5000,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,

        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,

        load_best_model_at_end=True,
        fp16=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    # 5. Inicjalizacja Trainera
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 6. Start treningu (z obsługą wznawiania)
    if resume_from:
        logger.info(f"*** Wznawianie treningu z checkpointu: {resume_from} ***")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        logger.info("*** Rozpoczynanie nowego treningu ***")
        trainer.train()

    # 7. Ewaluacja końcowa
    logger.info("Ewaluacja końcowa...")
    metrics = trainer.evaluate()

    # 8. Zapis wyników ewaluacji do pliku
    eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(eval_file, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info(f"Zapisano wyniki ewaluacji do: {eval_file}")

    # Zapisz finalny model
    final_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    return metrics


# 8. MAIN
def main():
    working_base = "relation_extraction_models"
    os.makedirs(working_base, exist_ok=True)

    models_to_test = {
        "RE_fine-tuned_BERT": "BC5CDR_fine-tuned_models/BERT-base",
        "RE_fine-tuned_BioBERT": "BC5CDR_fine-tuned_models/BioBERT",
        "RE_fine-tuned_BioClinicalBERT": "BC5CDR_fine-tuned_models/BioClinicalBERT",
        "RE_fine-tuned_PubMedBERT": "BC5CDR_fine-tuned_models/PubMedBERT"
    }

    print("Ładowanie zbioru danych BC5CDR...")
    dataset = load_dataset("bigbio/bc5cdr", name="bc5cdr_bigbio_kb", trust_remote_code=True)
    train_data = list(dataset["train"])
    test_data = list(dataset["test"])
    stats_file = os.path.join(working_base, "dataset_stats.txt")
    analyze_and_save_stats(train_data, test_data, stats_file)

    for new_model_name, source_path in models_to_test.items():
        if not os.path.exists(source_path):
            print(f"BŁĄD: Nie znaleziono modelu bazowego w ścieżce: {source_path}")
            print("Upewnij się, że folder 'BC5CDR_fine-tuned_models' jest w tym samym katalogu co skrypt.")
            continue

        current_output_dir = os.path.join(working_base, new_model_name)

        resume_checkpoint = None

        if os.path.isdir(current_output_dir):
            last_ckpt = get_last_checkpoint(current_output_dir)
            if last_ckpt:
                print(f"Znaleziono przerwany trening. Wznawianie z: {last_ckpt}")
                resume_checkpoint = last_ckpt
            else:
                print(f"Folder {current_output_dir} istnieje, ale nie zawiera checkpointów. Start od zera.")
        else:
            print(f"Folder docelowy nie istnieje. Tworzenie nowego treningu.")

        # Uruchomienie treningu
        metrics = train_re_model(
            model_path=source_path,
            train_data=train_data,
            test_data=test_data,
            output_dir=current_output_dir,
            resume_from=resume_checkpoint
        )

        print(f"\nPEŁNE WYNIKI DLA: {new_model_name}")
        for key, value in metrics.items():
            if key.startswith("eval_"):
                clean_key = key.replace("eval_", "")
                if isinstance(value, float):
                    print(f"{clean_key:<15}: {value:.4f}")
                else:
                    print(f"{clean_key:<15}: {value}")

        print("-" * 30)


if __name__ == "__main__":
    main()