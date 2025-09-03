"""
Mini fine-tuning de GPT-2 (español) con ~100 líneas de Don Quijote
------------------------------------------------------------------
• Pensado para Google Colab o local con GPU (también corre en CPU, más lento).
• Ajusta un modelo pequeño en español y genera un "cuento" de 5 oraciones.

Uso rápido (local):
  pip install transformers==4.45.2 datasets==2.20.0 accelerate==0.33.0 sentencepiece
  python gpt2_es_minift.py --epochs 2 --lr 1e-4 --block_size 128 --max_new_tokens 160

Referencias útiles:
• Guía de "Causal language modeling" de Transformers.  # ver README con citas

Autor: (tu nombre o institución)
Licencia del script: MIT (ajusta si lo prefieres)
"""

from __future__ import annotations
import argparse
import math
import os
import re
import sys
import textwrap
import urllib.request
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
    pipeline,
)


GUTENBERG_URL = "https://www.gutenberg.org/files/2000/2000-0.txt"  # Don Quijote (edición UTF-8)
DEFAULT_MODEL_ID = "datificate/gpt2-small-spanish"  # HF model card declara licencia Apache-2.0


def download_text(url: str, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def extract_quijote_100_lines(full_text: str) -> str:
    """
    Extrae ~100 líneas de la primera parte tras detectar un encabezado tipo 'CAPÍTULO I'.
    Robusto a acentos/mayúsculas.
    """
    lines = [l.strip() for l in full_text.splitlines()]
    start_idx = 0
    pattern = re.compile(r"CAP[ÍI]TULO\s+I\b", flags=re.IGNORECASE)
    for i, l in enumerate(lines):
        if pattern.search(l):
            start_idx = i
            break
    # Tomamos un rango razonable después del capítulo I, filtrando líneas muy cortas
    candidate = [l for l in lines[start_idx + 1 : start_idx + 800] if len(l.split()) > 3]
    if not candidate:
        # Fallback: tomar las primeras 400 líneas con contenido
        candidate = [l for l in lines if len(l.split()) > 3][:400]
    # Selección final de ~100 líneas
    selected = candidate[:100] if len(candidate) >= 100 else candidate
    return "\n".join(selected)


def build_datasets_from_text(
    text: str,
    multiplier: int = 5,
) -> Tuple[Dataset, Dataset]:
    """
    Crea datasets mínimos (train/test) duplicando un poco el texto para tener pasos de entrenamiento.
    """
    # Segmentos por párrafo o línea razonablemente larga
    paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) < 10:
        paragraphs = [l for l in text.split("\n") if len(l.split()) > 5]

    # Aumentamos datos por simple repetición (didáctico)
    train_texts = (paragraphs * multiplier)[:-max(1, len(paragraphs) // 5)] or paragraphs
    test_texts = paragraphs[:max(1, len(paragraphs) // 5)] or paragraphs[:1]

    train_ds = Dataset.from_dict({"text": train_texts})
    test_ds = Dataset.from_dict({"text": test_texts})
    return train_ds, test_ds


def tokenize_and_group(
    train_ds: Dataset,
    test_ds: Dataset,
    tokenizer,
    block_size: int,
):
    def tok(batch):
        return tokenizer(batch["text"])

    train_tok = train_ds.map(tok, batched=True, remove_columns=["text"])
    test_tok = test_ds.map(tok, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    train_lm = train_tok.map(group_texts, batched=True)
    test_lm = test_tok.map(group_texts, batched=True)

    return train_lm, test_lm


def cut_to_n_sentences(text: str, n: int = 5) -> str:
    """
    Recorta a n oraciones usando puntuación '. ! ?' (español/inglés) y evita cortes a mitad.
    """
    # Conserva signos finales al segmentar
    parts = re.split(r"([.!?])", text)
    out, acc = [], ""
    for i in range(0, len(parts) - 1, 2):
        acc = (acc + parts[i]).strip()
        end = parts[i + 1]
        if end in ".!?":
            acc = (acc + end).strip()
            if acc:
                out.append(acc)
            acc = ""
        if len(out) >= n:
            break
    if not out and text.strip():
        return text.strip()
    return " ".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Mini fine-tuning de GPT-2 (es) con 100 líneas de Don Quijote y generación de 5 oraciones."
    )
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="ID del modelo en Hugging Face Hub.")
    parser.add_argument("--data_url", type=str, default=GUTENBERG_URL, help="URL de texto fuente (Gutenberg).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de aleatoriedad.")
    parser.add_argument("--epochs", type=int, default=2, help="Épocas de entrenamiento.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--block_size", type=int, default=128, help="Tamaño de ventana/fragmento de tokens.")
    parser.add_argument("--train_bs", type=int, default=2, help="Batch por dispositivo (train).")
    parser.add_argument("--eval_bs", type=int, default=2, help="Batch por dispositivo (eval).")
    parser.add_argument("--grad_accum", type=int, default=4, help="Acumulación de gradientes.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Frecuencia de evaluación (steps).")
    parser.add_argument("--save_steps", type=int, default=200, help="Frecuencia de guardado (steps).")
    parser.add_argument("--output_dir", type=str, default="gpt2_es_quijote_minift", help="Directorio de salida.")
    parser.add_argument("--max_new_tokens", type=int, default=160, help="Tokens nuevos a generar.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Creatividad: mayor => más variedad.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Núcleo (top-p) sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Penalización repetición.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Completa un cuento de cinco oraciones, tono juvenil, "
            "sobre un estudiante que descubre un manuscrito antiguo en la biblioteca de su colegio. "
        ),
        help="Prompt de generación.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    print("Descargando texto desde:", args.data_url)
    full = download_text(args.data_url)
    dq100 = extract_quijote_100_lines(full)
    Path("don_quijote_100.txt").write_text(dq100, encoding="utf-8")
    print("Guardado don_quijote_100.txt con", len(dq100.splitlines()), "líneas.")

    print("Creando datasets…")
    train_ds, test_ds = build_datasets_from_text(dq100, multiplier=5)

    print("Cargando tokenizer y modelo:", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # GPT-2 no trae pad_token: usar eos como pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_lm, test_lm = tokenize_and_group(train_ds, test_ds, tokenizer, block_size=args.block_size)

    fp16 = torch.cuda.is_available()
    print("GPU disponible:", torch.cuda.is_available())

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        fp16=fp16,
        report_to=[],  # sin W&B
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_lm,
        eval_dataset=test_lm,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Entrenando…")
    trainer.train()

    print("Evaluando…")
    eval_res = trainer.evaluate()
    eval_loss = float(eval_res.get("eval_loss", float("nan")))
    print(f"Eval loss: {eval_loss:.4f}" if not math.isnan(eval_loss) else "Eval loss: NaN")
    if not math.isnan(eval_loss):
        try:
            ppl = math.exp(eval_loss)
            print(f"Perplejidad aproximada: {ppl:.2f}")
        except OverflowError:
            pass

    print("Generando cuento (5 oraciones)…")
    generator = pipeline(
        "text-generation",
        model=trainer.model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    gen_out = generator(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )[0]["generated_text"]

    # Mostrar solo lo nuevo y recortar a 5 oraciones
    only_new = gen_out[len(args.prompt) :]
    cuento_5 = cut_to_n_sentences(only_new, n=5)
    print("\n=== Cuento (5 oraciones) ===\n")
    print(textwrap.fill(cuento_5, width=100))

    print("\nGuardando modelo final…")
    save_dir = Path(args.output_dir) / "final"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print("Listo. Carpeta guardada en:", save_dir.resolve())


if __name__ == "__main__":
    main()