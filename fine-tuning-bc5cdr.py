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


# 3. Funkcja spłaszczania danych
def flatten_data_kb(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Przekształca zagnieżdżoną strukturę 'bc5cdr_bigbio_kb'
    na płaskie kolumny 'text' i 'entities'.
    """
    new_texts = []
    new_entities = []

    # Iterujemy po każdym dokumencie w batchu
    for i in range(len(batch.get("document_id", batch.get("pmid", [])))):
        passage_list = batch["passages"][i]

        try:
            passage_texts = [p["text"][0] for p in passage_list if p.get("text")]
        except IndexError:
            logger.warning(f"Błąd w passage dla dokumentu {i}, p['text'] może być puste.")
            passage_texts = []

        full_text = " ".join(passage_texts)
        all_entities = batch["entities"][i]
        new_texts.append(full_text)
        new_entities.append(all_entities)

    return {"text": new_texts, "entities": new_entities}


# 4. Funkcja przetwarzania danych
def convert_to_tokens_and_labels(
        samples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerFast,
        label_list_map: List[str]
) -> Dict[str, List[Any]]:
    """
    Przetwarza listę "spłaszczonych" próbek na sformatowane dane wejściowe.
    Używa strategii "pierwszego znaku" do mapowania etykiet.
    """
    label2id_map = {l: i for i, l in enumerate(label_list_map)}

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for sample in samples:
        text = sample["text"]
        entities = sample["entities"]

        # 1. Tworzenie mapowania na poziomie ZNAKÓW
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

        # 2. Tokenizacja
        tokenized = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512
        )
        offsets = tokenized['offset_mapping']
        labels = []

        # 3. Wyrównywanie etykiet ("first wordpiece strategy")
        for (start, end) in offsets:
            if start == end: # Tokeny specjalne [CLS], [SEP]
                labels.append(-100)
            else:
                # Weź etykietę z PIERWSZEGO znaku tego tokenu
                char_label_str = char_labels[start]
                labels.append(label2id_map.get(char_label_str, label2id_map['O']))

        all_input_ids.append(tokenized['input_ids'])
        all_attention_masks.append(tokenized['attention_mask'])
        all_labels.append(labels)

    return {"input_ids": all_input_ids, "attention_mask": all_attention_masks, "labels": all_labels}


# 5. Klasa Dataset
class NERDataset(TorchDataset):
    """
    Prosta klasa opakowująca (wrapper) dla PyTorch Dataset.
    """

    def __init__(self, encodings: Dict[str, List[Any]]):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


# 6. Główna funkcja treningu i ewaluacji
def train_and_eval(
        model_name: str,
        train_data: List[Dict[str, Any]],
        dev_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        label_list: List[str],
        output_dir: str,
        epochs: int = 10
) -> Dict[str, float]:
    """
    Główna funkcja zarządzająca cyklem życia modelu.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            f"Tokenizer {model_name} nie jest 'Fast'. Może brakować 'offset_mapping'."
        )

    id2label_map = {i: l for i, l in enumerate(label_list)}
    label2id_map = {l: i for i, l in enumerate(label_list)}

    logger.info("Tokenizacja danych...")
    train_encodings = convert_to_tokens_and_labels(train_data, tokenizer, label_list)
    dev_encodings = convert_to_tokens_and_labels(dev_data, tokenizer, label_list)
    test_encodings = convert_to_tokens_and_labels(test_data, tokenizer, label_list)

    train_dataset = NERDataset(train_encodings)
    dev_dataset = NERDataset(dev_encodings)
    test_dataset = NERDataset(test_encodings)

    logger.info("Ładowanie modelu...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label_map,
        label2id=label2id_map
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100
    )
    metric = evaluate.load("seqeval")

    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        """Oblicza metryki seqeval (P, R, F1, Acc) dla NER, w tym metryki dla jednostek."""
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

        final_results = {}
        final_results["overall_accuracy"] = results["overall_accuracy"]
        final_results["overall_precision"] = results["overall_precision"]
        final_results["overall_recall"] = results["overall_recall"]
        final_results["overall_f1"] = results["overall_f1"]

        for entity_type in entity_types:
            if entity_type in results:
                final_results[f"{entity_type}_precision"] = results[entity_type]["precision"]
                final_results[f"{entity_type}_recall"] = results[entity_type]["recall"]
                final_results[f"{entity_type}_f1"] = results[entity_type]["f1"]
                final_results[f"{entity_type}_number"] = results[entity_type]["number"]

        return final_results

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
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
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Wznawianie treningu
    output_dir_check = training_args.output_dir
    last_checkpoint = None

    if os.path.isdir(output_dir_check):
        logger.info(f"Sprawdzanie istnienia checkpointów w: {output_dir_check}")
        checkpoint_dirs = []
        for item in os.listdir(output_dir_check):
            item_path = os.path.join(output_dir_check, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                checkpoint_dirs.append(item)

        if checkpoint_dirs:
            try:
                step_numbers = [int(d.split('-')[-1]) for d in checkpoint_dirs]
                latest_step = max(step_numbers)
                last_checkpoint = os.path.join(output_dir_check, f"checkpoint-{latest_step}")
                logger.info(f"Znaleziono ostatni checkpoint: {last_checkpoint}")
            except ValueError:
                logger.warning(f"Nie udało się odczytać numerów checkpointów w {output_dir_check}. Zaczynam od zera.")

    if last_checkpoint:
        logger.info(f"Wznawianie treningu z: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info(f"Brak checkpointu. Rozpoczynanie nowego treningu.")
        trainer.train()

    logger.info("Ewaluacja na zbiorze testowym...")
    results = trainer.evaluate(test_dataset, metric_key_prefix="eval")

    logger.info("Zapisywanie modelu...")
    trainer.save_model(output_dir)
    return results


# 7. Funkcja główna
def main() -> None:
    """
        Główny punkt wejścia skryptu.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    output_base = os.path.join(base_dir, "fine-tuning_outputs")
    os.makedirs(output_base, exist_ok=True)

    models = {
        "BERT-base": "bert-base-uncased",
        "BioBERT": "dmis-lab/biobert-base-cased-v1.1", #jest "uncased" mimo nazwy
        "BioClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
        "PubMedBERT": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    }

    logger.info("Wczytywanie zbioru danych BC5CDR z Hugging Face...")
    dataset = load_dataset(
        "bigbio/bc5cdr",
        name="bc5cdr_bigbio_kb",
        trust_remote_code=True
    )
    logger.info(f"Załadowano zbiór (surowy): {dataset}")

    logger.info("Spłaszczanie danych...")
    flattened_dataset = dataset.map(
        flatten_data_kb,
        batched=True,
        num_proc=1,
        desc="Flattening KB data"
    )
    logger.info(f"Zbiór po spłaszczeniu: {flattened_dataset}")

    logger.info("Konwertowanie zbiorów danych na listy...")
    train_data: List[Dict[str, Any]] = list(flattened_dataset["train"])
    dev_data: List[Dict[str, Any]] = list(flattened_dataset["validation"])
    test_data: List[Dict[str, Any]] = list(flattened_dataset["test"])
    logger.info(f"Rozmiary zbiorów: Trening={len(train_data)}, Walidacja={len(dev_data)}, Test={len(test_data)}")

    results_table = {}
    for name, model_name in models.items():
        logger.info(f"Rozpoczynanie pracy z modelem: {name}")
        output_model_dir = os.path.join(output_base, name)
        os.makedirs(output_model_dir, exist_ok=True)

        # TRENING
        results = train_and_eval(
            model_name=model_name,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            label_list=label_list,
            output_dir=output_model_dir,
            epochs=11
        )

        final_metrics = {
            k.replace("eval_", ""): v
            for k, v in results.items()
            if k.startswith("eval_")
        }

        results_table[name] = final_metrics
        logger.info(f"Wyniki (test) dla {name}: {final_metrics}")
        logger.info(f"Zakończono pracę z modelem: {name}")

    print("\n\n" + "=" * 80)
    print("--- SZCZEGÓŁOWE PORÓWNANIE WYNIKÓW (zbiór testowy BC5CDR) ---")
    print("=" * 80)

    for model, res in results_table.items():
        print(f"\nModel: {model}")
        print("-" * 80)

        print(f"    Ogólne (Overall):   "
              f"F1: {res.get('overall_f1', 0.0):.6f}, "
              f"Precyzja: {res.get('overall_precision', 0.0):.6f}, "
              f"Czułość: {res.get('overall_recall', 0.0):.6f}, "
              f"Dokładność: {res.get('overall_accuracy', 0.0):.6f}")

        print(f"    Choroby (Disease):  "
              f"F1: {res.get('Disease_f1', 0.0):.6f}, "
              f"Precyzja: {res.get('Disease_precision', 0.0):.6f}, "
              f"Czułość: {res.get('Disease_recall', 0.0):.6f}")

        print(f"    Związki chemiczne (Chemical): "
              f"F1: {res.get('Chemical_f1', 0.0):.6f}, "
              f"Precyzja: {res.get('Chemical_precision', 0.0):.6f}, "
              f"Czułość: {res.get('Chemical_recall', 0.0):.6f}")

        print("=" * 80)

if __name__ == "__main__":
    main()