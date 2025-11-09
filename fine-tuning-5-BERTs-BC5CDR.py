import os
import logging
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
    EvalPrediction
)
import evaluate
from datasets import load_dataset, Dataset as HuggingFaceDataset
from torch.utils.data import Dataset as TorchDataset
from typing import List, Dict, Any
from torch import Tensor

# --- logger ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Definicja etykiet dla BC5CDR ---
entity_types = ["Chemical", "Disease"]
label_list = ['O'] + [f'B-{et}' for et in entity_types] + [f'I-{et}' for et in entity_types]


# --- FUNKCJA: spłaszczanie zagnieżdżonych danych BC5CDR ---
def flatten_bc5cdr_passages(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Przekształca zagnieżdżoną strukturę 'passages' z BC5CDR (konfiguracja 'source')
    na płaskie kolumny 'text' i 'entities'.

    Ta funkcja jest przeznaczona do użycia z `dataset.map(batched=True)`.

    Wejście (batch):
        Słownik (batch) z Hugging Face `datasets`, np.:
        {'passages': [
            [{'text': 'Tytuł...', 'entities': [...]}, {'text': 'Abstrakt...', 'entities': [...]}]
        , ...]}

    Wyjście:
        Słownik z nowymi, płaskimi kolumnami:
        {'text': ['Tytuł Abstrakt...'], 'entities': [[...], ...]}
    """
    new_texts = []
    new_entities = []

    # Iterujemy po każdym wierszu ('passages') w batchu
    for passage_list in batch["passages"]:
        full_text = ""
        all_entities = []
        current_offset = 0

        # Iterujemy po fragmentach (zwykle 'title' i 'abstract')
        for i, passage in enumerate(passage_list):
            passage_text = passage["text"]
            if isinstance(passage_text, list):  # Zabezpieczenie, czasem to lista słów
                passage_text = " ".join(passage_text)

            # Dodajemy spację między tytułem a abstraktem
            if i > 0:
                full_text += " "
                current_offset += 1

            full_text += passage_text

            # Przesuwamy offsety encji z bieżącego fragmentu
            for entity in passage.get("entities", []):  # Użyj .get dla bezpieczeństwa
                ent = {"type": entity["type"], "offsets": []}
                for (start, end) in entity["offsets"]:
                    # Dodajemy bieżący offset (długość poprzednich tekstów)
                    ent["offsets"].append([start + current_offset, end + current_offset])
                all_entities.append(ent)

            # Aktualizujemy offset dla następnego fragmentu
            current_offset = len(full_text)

        new_texts.append(full_text)
        new_entities.append(all_entities)

    return {"text": new_texts, "entities": new_entities}


# --- KONWERSJA NA TOKENY I ETYKIETY ---
def convert_to_tokens_and_labels(
        samples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerFast,
        label_list_map: List[str]
) -> Dict[str, List[Any]]:
    """
    Przetwarza listę "spłaszczonych" próbek na sformatowane dane wejściowe dla modelu.

    Dla każdej próbki:
    1. Tworzy mapowanie etykiet na poziomie znaków (char_labels).
    2. Tokenizuje tekst (z `offset_mapping`).
    3. Wyrównuje (align) etykiety znakowe do tokenów w schemacie B-I-O.

    Wejście (samples):
        Lista słowników (wynik funkcji `flatten_bc5cdr_passages`)

    Wyjście:
        Słownik list gotowy do użycia przez NERDataset
    """
    label2id_map = {l: i for i, l in enumerate(label_list_map)}
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for sample in samples:
        text = sample["text"]
        entities = sample["entities"]

        # 1. Tworzenie mapowania na poziomie znaków
        char_labels = ['O'] * len(text)
        for entity in entities:
            entity_type = entity['type']
            for (start, end) in entity['offsets']:
                if start < len(text) and end <= len(text):
                    char_labels[start] = f"B-{entity_type}"
                    for i in range(start + 1, end):
                        if i < len(text):  # Dodatkowe zabezpieczenie
                            char_labels[i] = f"I-{entity_type}"

        # 2. Tokenizacja
        tokenized = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            # Padding zostanie dodany przez DataCollator
        )
        offsets = tokenized['offset_mapping']
        labels = []

        # 3. Wyrównywanie etykiet
        for (start, end) in offsets:
            if start == end:  # Tokeny specjalne ([CLS], [SEP], [PAD])
                labels.append(-100)  # -100 jest ignorowane przez loss function
            else:
                # Użyj etykiety pierwszego znaku tokenu
                char_label = char_labels[start]
                labels.append(label2id_map.get(char_label, label2id_map['O']))

        all_input_ids.append(tokenized['input_ids'])
        all_attention_masks.append(tokenized['attention_mask'])
        all_labels.append(labels)

    return {"input_ids": all_input_ids, "attention_mask": all_attention_masks, "labels": all_labels}


# --- Dataset dla Hugging Face Trainer ---
class NERDataset(TorchDataset):
    """
    Prosta klasa opakowująca (wrapper) dla PyTorch Dataset.
    Przechowuje przetworzone dane (encodings) i zwraca je jako tensory.
    """

    def __init__(self, encodings: Dict[str, List[Any]]):
        """Inicjalizuje Dataset."""
        self.encodings = encodings

    def __len__(self) -> int:
        """Zwraca całkowitą liczbę próbek."""
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Pobiera pojedynczą próbkę i konwertuje ją na tensory PyTorch."""
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


# --- Trening i ewaluacja ---
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

    1. Inicjalizuje tokenizer.
    2. Przetwarza surowe dane (listy dict) na obiekty NERDataset.
    3. Inicjalizuje model AutoModelForTokenClassification.
    4. Definiuje metryki (seqeval).
    5. Inicjalizuje i uruchamia `Trainer`.
    6. Ewaluuje model na zbiorze testowym.
    7. Zwraca wyniki ewaluacji.

    Wejście:
        model_name (str): Nazwa modelu z Hugging Face Hub.
        train_data/dev_data/test_data (List[dict]): Listy próbek (spłaszczone).
        label_list (List[str]): Globalna lista etykiet (np. ['O', 'B-Disease', ...]).
        output_dir (str): Ścieżka do zapisu modelu i logów.
        epochs (int): Liczba epok treningu.

    Wyjście:
        Słownik (dict) zawierający metryki ewaluacji na zbiorze testowym.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Upewnij się, że tokenizer jest "szybki" (Fast), aby mieć 'offset_mapping'
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        logger.warning(
            f"Tokenizer {model_name} nie jest 'Fast'. Może brakować 'offset_mapping'."
        )

    id2label_map = {i: l for i, l in enumerate(label_list)}
    label2id_map = {l: i for i, l in enumerate(label_list)}

    train_encodings = convert_to_tokens_and_labels(train_data, tokenizer, label_list)
    dev_encodings = convert_to_tokens_and_labels(dev_data, tokenizer, label_list)
    test_encodings = convert_to_tokens_and_labels(test_data, tokenizer, label_list)

    train_dataset = NERDataset(train_encodings)
    dev_dataset = NERDataset(dev_encodings)
    test_dataset = NERDataset(test_encodings)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label_map,
        label2id=label2id_map
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = evaluate.load("seqeval")

    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        """Oblicza metryki seqeval (P, R, F1, Acc) dla NER."""
        preds = np.argmax(p.predictions, axis=2)
        labels = p.label_ids

        # Odtwórz etykiety tekstowe i usuń tokeny [-100]
        true_preds = [
            [label_list[p] for (p, l) in zip(prediction, label_row) if l != -100]
            for prediction, label_row in zip(preds, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label_row) if l != -100]
            for prediction, label_row in zip(preds, labels)
        ]

        results = metric.compute(predictions=true_preds, references=true_labels, zero_division=0)
        return {
            "accuracy": results["overall_accuracy"],
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"]
        }

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
        metric_for_best_model="f1",
        report_to="none",
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

    logger.info("Rozpoczynanie treningu...")
    trainer.train()

    logger.info("Ewaluacja na zbiorze testowym...")
    results = trainer.evaluate(test_dataset, metric_key_prefix="eval")

    logger.info("Zapisywanie modelu...")
    trainer.save_model(output_dir)
    return results


def main() -> None:
    """
    Główny punkt wejścia skryptu.

    1. Definiuje listę modeli do przetestowania.
    2. Wczytuje zagnieżdżony zbiór danych BC5CDR ('source').
    3. Spłaszcza zbiór danych (łączy tytuł i abstrakt).
    4. Konwertuje zbiory danych na listy.
    5. Iteruje przez listę modeli, wywołując `train_and_eval` dla każdego z nich.
    6. Zbiera wyniki i drukuje tabelę porównawczą.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    output_base = os.path.join(base_dir, "bc5cdr_outputs")
    os.makedirs(output_base, exist_ok=True)

    models = {
        "BERT-base": "bert-base-uncased",
        "BioBERT": "dmis-lab/biobert-base-cased-v1.S",
        "BioClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
        "BioMedBERT": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "SapBERT": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    }

    logger.info("Wczytywanie zbioru danych BC5CDR z Hugging Face...")
    # Używamy 'bc5cdr_source', aby ręcznie pokazać proces spłaszczania
    dataset = load_dataset("bigbio/bc5cdr", name="bc5cdr_source", trust_remote_code=True)
    logger.info(f"Załadowano zbiór: {dataset}")

    logger.info("Spłaszczanie zagnieżdżonych danych BC5CDR (łączenie tytułu i abstraktu)...")
    flattened_dataset = dataset.map(
        flatten_bc5cdr_passages,
        batched=True,
        num_proc=1,  # Bezpieczniej dla Windows i debugowania
        desc="Flattening passages"
    )
    logger.info(f"Zbiór po spłaszczeniu: {flattened_dataset}")

    logger.info("Konwertowanie zbiorów danych na listy...")
    train_data: List[Dict[str, Any]] = list(flattened_dataset["train"])
    dev_data: List[Dict[str, Any]] = list(flattened_dataset["validation"])
    test_data: List[Dict[str, Any]] = list(flattened_dataset["test"])
    logger.info(f"Rozmiary zbiorów: Trening={len(train_data)}, Walidacja={len(dev_data)}, Test={len(test_data)}")

    results_table = {}
    for name, model_name in models.items():
        logger.info(f"--- Rozpoczynanie pracy z modelem: {name} ({model_name}) ---")
        output_model_dir = os.path.join(output_base, name)
        os.makedirs(output_model_dir, exist_ok=True)

        results = train_and_eval(
            model_name=model_name,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            label_list=label_list,
            output_dir=output_model_dir,
            epochs=8
        )

        final_metrics = {
            "accuracy": results["eval_accuracy"],
            "precision": results["eval_precision"],
            "recall": results["eval_recall"],
            "f1": results["eval_f1"]
        }

        results_table[name] = final_metrics
        logger.info(f"Wyniki (test) dla {name}: {final_metrics}")
        logger.info(f"--- Zakończono pracę z modelem: {name} ---")

    print("\n\n" + "=" * 50)
    print("--- PORÓWNANIE WYNIKÓW (zbiór testowy BC5CDR) ---")
    print("=" * 50)
    print(f"{'Model':20} {'Dokładność':10} {'Precyzja':10} {'Czułość':10} {'Wynik F1':10}")
    print("-" * 60)
    for model, res in results_table.items():
        print(f"{model:20} {res['accuracy']:.6f} {res['precision']:.6f} {res['recall']:.6f} {res['f1']:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()