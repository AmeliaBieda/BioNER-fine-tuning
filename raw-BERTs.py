import os
import torch
import math
import logging
import warnings
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# --- 0. Konfiguracja logowania i ostrzeżeń ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Użyj GPU, jeśli jest dostępne
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Używane urządzenie: {DEVICE}")


# --- 1. Implementacja funkcji ładowania danych (Brakujący element) ---

def load_pubtator_file(filepath):
    """
    Parsuje plik w formacie PubTator i zwraca listę rekordów.
    Każdy rekord to słownik zawierający połączony tekst (tytuł + abstrakt).
    """
    records = []
    current_pmid = None
    current_text = ""

    logger.info(f"Próba załadowania pliku: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Pusta linia oznacza koniec rekordu
                if not line:
                    if current_pmid:
                        records.append({'pmid': current_pmid, 'text': current_text})
                    current_pmid = None
                    current_text = ""
                    continue

                parts = line.split('|')

                # Linie tekstu (tytuł lub abstrakt)
                if len(parts) >= 3 and parts[1] in ('t', 'a'):
                    pmid = parts[0]
                    text_part = parts[2]

                    if pmid != current_pmid:
                        # Zapisz poprzedni rekord (jeśli istniał)
                        if current_pmid:
                            records.append({'pmid': current_pmid, 'text': current_text})

                        # Rozpocznij nowy rekord
                        current_pmid = pmid
                        current_text = text_part
                    else:
                        # Dołącz abstrakt do tytułu
                        current_text += " " + text_part

                # Ignoruj linie z adnotacjami (np. 12345\tstart\tend...)

            # Dodaj ostatni przetwarzany rekord
            if current_pmid:
                records.append({'pmid': current_pmid, 'text': current_text})

    except FileNotFoundError:
        logger.error(f"KRYTYCZNY BŁĄD: Nie znaleziono pliku: {filepath}")
        return []
    except Exception as e:
        logger.error(f"Błąd podczas czytania pliku {filepath}: {e}")
        return []

    return records


# --- 2. Ładowanie danych (Twoja struktura plików) ---

# UWAGA: os.path.dirname(__file__) działa tylko przy uruchamianiu jako plik .py
# Jeśli używasz Jupyter Notebook, zastąp to kropką:
# SCRIPT_DIR = "."
try:
    SCRIPT_DIR = os.path.dirname(__file__)
except NameError:
    SCRIPT_DIR = "."  # Awaryjnie dla środowisk typu notebook
    logger.warning("Uruchamiasz w środowisku interaktywnym. Ustawiam bazowy katalog na '.'")

base_dir = os.path.join(SCRIPT_DIR, "CDR_Data", "CDR.Corpus.v010516")
train_path = os.path.join(base_dir, "CDR_TrainingSet.PubTator.txt")
# Poniższe pliki nie są nam potrzebne do ewaluacji MLM,
# ale możesz je dodać, jeśli chcesz mieć większy korpus
# dev_path = os.path.join(base_dir, "CDR_DevelopmentSet.PubTator.txt")
# test_path = os.path.join(base_dir, "CDR_TestSet.PubTator.txt")

# Ładujemy tylko dane treningowe (zazwyczaj wystarczająco reprezentatywne)
train_data_raw = load_pubtator_file(train_path)

if not train_data_raw:
    logger.error("Nie udało się załadować danych. Przerwanie skryptu.")
    exit()

logger.info(f"Załadowano {len(train_data_raw)} rekordów treningowych.")

# Konwertujemy listę słowników na obiekt Dataset z Hugging Face
# Potrzebujemy tylko kolumny z tekstem
text_list = [record['text'] for record in train_data_raw]
full_dataset = Dataset.from_dict({'text_data': text_list})

# Dla celów demonstracyjnych i szybkości, weźmy mniejszy podzbiór
# Zwiększ tę liczbę lub usuń .select() aby testować na pełnym zbiorze
try:
    subset_dataset = full_dataset.select(range(min(2000, len(full_dataset))))
    logger.info(f"Używam podzbioru {len(subset_dataset)} próbek do ewaluacji.\n")
except ValueError:
    subset_dataset = full_dataset
    logger.info(f"Używam pełnego zbioru {len(subset_dataset)} próbek do ewaluacji.\n")

# --- 3. Konfiguracja ewaluacji (MLM Perplexity) ---

# Modele, które chcemy porównać
'''MODEL_CHECKPOINTS = {
    "BERT": "bert-base-cased",
    "SciBERT": "allenai/scibert_scivocab_cased",
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1"
}
'''
MODEL_CHECKPOINTS = {
        "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
            #"microsoft/BiomedNLP-BioBERT-Base-v1.1", "monologg/biobert_v1.1_pubmed", "dmis-lab/biobert-v1.1-cased", ,
        #"BioClinicalBERT-uncased": "emilyalsentzer/Bio_ClinicalBERT",
        "SapBERT": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "SciBERT": "allenai/scibert_scivocab_cased",
        "BERT": "bert-base-cased"
    }
# Twój folder wyjściowy
output_base = os.path.join(SCRIPT_DIR, "outputs_perplexity")
os.makedirs(output_base, exist_ok=True)

# Słownik na wyniki
evaluation_results = {}

# --- 4. Pętla ewaluacyjna ---

for model_name, checkpoint in MODEL_CHECKPOINTS.items():
    logger.info(f"Ewaluacja modelu: {model_name} ({checkpoint})")

    # 4.1. Załaduj tokenizer i model (dla Masked Language Modeling)
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(DEVICE)
    except Exception as e:
        logger.error(f"Nie udało się załadować modelu lub tokenizera: {e}")
        continue


    # 4.2. Funkcja tokenizująca
    def tokenize_function(examples):
        return tokenizer(
            examples["text_data"],
            truncation=True,
            padding=False,
            max_length=512,
            return_special_tokens_mask=True  # Potrzebne dla DataCollator
        )


    # 4.3. Tokenizuj zbiór danych
    logger.info("Tokenizacja danych dla bieżącego modelu...")
    tokenized_dataset = subset_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=subset_dataset.column_names
    )

    # 4.4. Data Collator (automatyczne maskowanie tokenów)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # 4.5. Konfiguracja Trenera (używamy go tylko do ewaluacji)
    output_dir = os.path.join(output_base, f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_eval")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=8,
        do_train=False,
        do_eval=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=tokenized_dataset
    )

    # 4.6. Uruchom ewaluację
    logger.info("Uruchamianie ewaluacji...")
    eval_metrics = trainer.evaluate()

    # 4.7. Oblicz Perplexity
    eval_loss = eval_metrics["eval_loss"]
    perplexity = math.exp(eval_loss)

    evaluation_results[model_name] = {
        "model_checkpoint": checkpoint,
        "loss": eval_loss,
        "perplexity": perplexity
    }

    logger.info(f"Wynik dla {model_name}: Loss = {eval_loss:.4f}, Perplexity = {perplexity:.4f}\n")

# --- 5. Podsumowanie wyników ---

print("\n\n" + "=" * 80)
print("--- 📊 Końcowe podsumowanie (posortowane od najlepszego) ---")
print("=" * 80)
print(f"{'Model':<18} | {'Loss (↓)':<10} | {'Perplexity (↓)':<15} | {'Checkpoint'}")
print("-" * 80)

# Sortuj wyniki od najniższej (najlepszej) perpleksji
sorted_results = sorted(evaluation_results.items(), key=lambda item: item[1]['perplexity'])

for model_name, res in sorted_results:
    print(f"{model_name:<18} | {res['loss']:<10.4f} | {res['perplexity']:<15.4f} | {res['model_checkpoint']}")

print("\n**Kluczowa interpretacja:**")
print("Model z najniższą wartością **Perplexity** (perpleksja) jest najlepiej dopasowany")
print("do języka używanego w Twoich plikach BC5CDR *przed* jakimkolwiek fine-tuningiem.")
print("\nOczekiwane wyniki: BioBERT i SciBERT powinny mieć znacznie niższą perpleksję niż standardowy BERT.")