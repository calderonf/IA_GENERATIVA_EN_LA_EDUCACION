# ==== Mini fine-tuning GPT-2 (ES) | Versión simple para Colab ====
# - Edita la sección CONFIG y ejecuta.
# - Al final se entrena y genera automáticamente.
# - En otra celda puedes llamar: generate_text("tu prompt...") para reutilizar el modelo guardado.

from __future__ import annotations
import math, re, textwrap, urllib.request
from pathlib import Path
from typing import Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
    set_seed, pipeline
)

# ---------------------------
# CONFIG: edita aquí si quieres
# ---------------------------
MODEL_ID            = "datificate/gpt2-small-spanish"
DATA_URL            = "https://www.gutenberg.org/files/2000/2000-0.txt"   # Don Quijote (UTF-8)
SEED                = 42
EPOCHS              = 2
LR                  = 1e-4
BLOCK_SIZE          = 128
TRAIN_BS            = 2
EVAL_BS             = 2
GRAD_ACCUM          = 4
WEIGHT_DECAY        = 0.01
OUTPUT_DIR          = "gpt2_es_quijote_minift"   # ./gpt2_es_quijote_minift/final
N_LINES             = 100                        # cuántas líneas extraer del texto
REPEAT_MULTIPLIER   = 5                          # “aumento” didáctico por repetición

# Parámetros de generación por defecto (puedes cambiarlos aquí o al llamar generate_text)
MAX_NEW_TOKENS      = 160
TEMPERATURE         = 0.9
TOP_K               = 50
TOP_P               = 0.95
REPETITION_PENALTY  = 1.1
DEFAULT_PROMPT = (
    "Completa un cuento de cinco oraciones, tono juvenil, "
    "sobre un estudiante que descubre un manuscrito antiguo en la biblioteca de su colegio. "
)

# ---------------------------
# Utilidades de datos
# ---------------------------
def download_text(url: str, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")

def extract_quijote_lines(full_text: str, n: int = 100) -> str:
    """Extrae ~n líneas a partir del capítulo primero (robusto a acentos)."""
    lines = [l.strip() for l in full_text.splitlines()]
    pattern = re.compile(
        r"Cap[íi]tulo\s+primero\.?\s*De\s+lo\s+que\s+el\s+cura\s+y\s+el\s+barbero",
        flags=re.IGNORECASE | re.UNICODE
    )
    start_idx = 0
    for i, l in enumerate(lines):
        if pattern.search(l):
            start_idx = i
            break
    candidate = [l for l in lines[start_idx+1:start_idx+800] if len(l.split()) > 3]
    if not candidate:  # fallback si cambia la edición
        candidate = [l for l in lines if len(l.split()) > 3][:400]
    selected = candidate[:n] if len(candidate) >= n else candidate
    return "\n".join(selected)

def build_datasets_from_text(text: str, multiplier: int = 5) -> Tuple[Dataset, Dataset]:
    """Crea datasets mínimos de train/test (aumenta por repetición para tener más pasos)."""
    paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) < 10:
        paragraphs = [l for l in text.split("\n") if len(l.split()) > 5]
    train_texts = (paragraphs * multiplier)[:-max(1, len(paragraphs)//5)] or paragraphs
    test_texts  =  paragraphs[:max(1, len(paragraphs)//5)] or paragraphs[:1]
    return Dataset.from_dict({"text": train_texts}), Dataset.from_dict({"text": test_texts})

def tokenize_and_group(train_ds: Dataset, test_ds: Dataset, tokenizer, block_size: int):
    def tok(batch): return tokenizer(batch["text"])
    train_tok = train_ds.map(tok, batched=True, remove_columns=["text"])
    test_tok  =  test_ds.map(tok,  batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return train_tok.map(group_texts, batched=True), test_tok.map(group_texts, batched=True)


# ---------------------------
# Entrenamiento + guardado
# ---------------------------
def train_and_save():
    """Entrena rápidamente y guarda en <OUTPUT_DIR>/final. Devuelve (tokenizer, trainer)."""
    set_seed(SEED)
    print("Descargando texto…", DATA_URL)
    full = download_text(DATA_URL)
    dq = extract_quijote_lines(full, n=N_LINES)
    Path("don_quijote_100.txt").write_text(dq, encoding="utf-8")
    print("Guardado don_quijote_100.txt con", len(dq.splitlines()), "líneas.")

    train_ds, test_ds = build_datasets_from_text(dq, multiplier=REPEAT_MULTIPLIER)

    print("Cargando tokenizer y modelo:", MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_lm, test_lm = tokenize_and_group(train_ds, test_ds, tok, block_size=BLOCK_SIZE)

    print("GPU disponible:", torch.cuda.is_available())
    targs = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        logging_steps=10,
        do_eval=False,             # simple y robusto durante train
        save_steps=200,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.resize_token_embeddings(len(tok))
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_lm,
        data_collator=collator,
        tokenizer=tok,
    )

    print("Entrenando…")
    trainer.train()

    # Eval sencilla tras entrenar
    print("Evaluando…")
    eval_res = trainer.evaluate(eval_dataset=test_lm)
    loss = float(eval_res.get("eval_loss", float("nan")))
    print("Eval loss:", f"{loss:.4f}" if not math.isnan(loss) else "NaN")
    if not math.isnan(loss):
        try:
            print("Perplejidad ~", math.exp(loss))
        except OverflowError:
            pass

    # Guardar
    save_dir = Path(OUTPUT_DIR) / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tok.save_pretrained(str(save_dir))
    print("Modelo guardado en:", save_dir.resolve())

    return tok, trainer

# ---------------------------
# Generación (reutilizable en otra celda)
# ---------------------------
_DEF_GENERATOR = None  # caché simple

def _get_generator_from_trainer(trainer, tokenizer):
    global _DEF_GENERATOR
    _DEF_GENERATOR = pipeline(
        "text-generation",
        model=trainer.model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    return _DEF_GENERATOR

def load_generator_from_disk(model_dir: str = f"{OUTPUT_DIR}/final"):
    """Carga un pipeline desde el directorio guardado (útil otro día sin reentrenar)."""
    print("Cargando generador desde:", Path(model_dir).resolve())
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForCausalLM.from_pretrained(model_dir)
    gen = pipeline("text-generation", model=mdl, tokenizer=tok,
                   device=0 if torch.cuda.is_available() else -1)
    return gen

def generate_text(
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,
    five_sentences: bool = True,
    generator=None,
):
    """Genera texto con el pipeline disponible; si no hay, carga desde disco."""
    global _DEF_GENERATOR
    gen = generator or _DEF_GENERATOR
    if gen is None:
        gen = load_generator_from_disk()  # intenta desde disco

    print("Prompt:\n", prompt)
    out = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        pad_token_id=gen.tokenizer.eos_token_id,
    )[0]["generated_text"]

    only_new = out[len(prompt):]
    print("\n=== Generación ===\n")
    print(textwrap.fill(only_new, width=100))
    return only_new

# ---------------------------
# Ejecuta automáticamente (como antes)
# ---------------------------
if __name__ == "__main__":
    tok, trainer = train_and_save()
    gen = _get_generator_from_trainer(trainer, tok)
    # Generación demo con los parámetros por defecto
    generate_text(generator=gen)

# Si estas en google colab puede llamar correr en otras celdas ejemplos reutilizando el modelo ya guardado así:
"""
gen = load_generator_from_disk()
generate_text("Escribe cinco oraciones sobre un colegio de curas quijotescos.", generator=gen)
"""