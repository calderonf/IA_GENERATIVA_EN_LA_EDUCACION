### gpt2\_es\_minift.py

```python
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
```

---

### README.md

````markdown
# Mini fine-tuning de GPT-2 (español) con ~100 líneas de *Don Quijote*

> Ajuste educativo **ultra-ligero** de un GPT-2 pequeño en español para **completar un cuento de 5 oraciones**.  
> Diseñado para **Google Colab** (GPU recomendada) o ejecución local.

---

## 🧭 Qué hace este repo

1. Descarga **100 líneas** de *Don Quijote* (dominio público) desde Project Gutenberg. :contentReference[oaicite:0]{index=0}  
2. Prepara un **dataset mínimo** (solo para demostración didáctica).  
3. Ajusta el modelo **`datificate/gpt2-small-spanish`** (licencia Apache-2.0 según su model card). :contentReference[oaicite:1]{index=1}  
4. Genera un **cuento de 5 oraciones** con sampling controlado.  
5. Guarda el modelo ajustado en `gpt2_es_quijote_minift/final/`.

> Para los curiosos: el procedimiento sigue la **receta oficial** de *Causal Language Modeling* de Transformers (sección “Language modeling”). :contentReference[oaicite:2]{index=2}

---

## ⚡️ Ejecución en Google Colab (recomendada)

1. **Abrir Colab** y activar GPU: `Entorno de ejecución → Cambiar tipo de entorno → GPU`.
2. Crear una celda e instalar dependencias:
   ```bash
   pip install "transformers==4.45.2" "datasets==2.20.0" "accelerate==0.33.0" sentencepiece
````

3. **Sube** `gpt2_es_minift.py` al espacio de trabajo de Colab.
4. Ejecuta:

   ```bash
   python gpt2_es_minift.py --epochs 2 --lr 1e-4 --block_size 128 --max_new_tokens 160
   ```

**Salida esperada (resumen):**

* Métrica `eval_loss` y su **perplejidad aproximada**.
* Un **cuento de 5 oraciones** con rasgos estilísticos del corpus breve (puede sonar arcaizante).

---

## 🖥️ Ejecución local

Requisitos:

* Python **3.9+** (probado en 3.10)
* GPU NVIDIA (opcional) con drivers/CUDA configurados

```bash
python -m venv .venv && source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install "transformers==4.45.2" "datasets==2.20.0" "accelerate==0.33.0" sentencepiece
python gpt2_es_minift.py
```

---

## 🔧 Parámetros útiles (CLI)

| Flag                   | Default                         | Explicación “amigable”                                                      |
| ---------------------- | ------------------------------- | --------------------------------------------------------------------------- |
| `--epochs`             | `2`                             | **Vueltas al cuaderno**; con pocos datos, 1–3 suelen bastar.                |
| `--lr`                 | `1e-4`                          | **Qué tan rápido corrijo**; muy grande = inestable, muy pequeño = lento.    |
| `--block_size`         | `128`                           | **Ventana de contexto** por lote; subirlo consume más VRAM.                 |
| `--train_bs`           | `2`                             | Tamaño de batch (train) por dispositivo.                                    |
| `--grad_accum`         | `4`                             | **Acumulación de gradientes** para simular batches más grandes.             |
| `--max_new_tokens`     | `160`                           | Largo máximo del texto **nuevo** a generar.                                 |
| `--temperature`        | `0.9`                           | Creatividad: ↑ temperatura ⇒ ↑ variedad.                                    |
| `--top_k` / `--top_p`  | `50` / `0.95`                   | Sampling controlado: **suerte con cinturón de seguridad**.                  |
| `--repetition_penalty` | `1.1`                           | Penaliza repeticiones molestas.                                             |
| `--model_id`           | `datificate/gpt2-small-spanish` | Puedes cambiar a otro GPT-2 en español si lo prefieres. ([Hugging Face][1]) |

---

## 🧪 ¿Qué aprendizaje ofrece?

* **Transferencia de estilo** con **muy pocos datos**: verás vocabulario y cadencia “quijotesca”.
* **Limitaciones**: el dataset diminuto propicia **sobreajuste**; no es un experimento de producción.
* **Buena práctica**: compara generaciones **antes** y **después** del ajuste y pide a tus estudiantes identificar “huellas” del corpus.

Para extender:

* Mezcla textos contemporáneos para un tono más moderno.
* Incrementa `epochs` y reduce `lr` para un ajuste más fino (si tienes más datos).
* Explora kernels/optimizaciones modernas si escalas (p. ej., *FlashAttention*). ([arXiv][2], [ar5iv][3], [GitHub][4])

---

## 🧰 Estructura de archivos

```
.
├── gpt2_es_minift.py          # Script principal (entrena, evalúa, genera y guarda)
├── don_quijote_100.txt        # Se crea automáticamente (100 líneas extraídas)
└── gpt2_es_quijote_minift/
    └── final/                 # Carpeta del modelo ajustado + tokenizer
```

---

## 🧑‍🏫 Consejos docentes (10 min, sin código)

**Actividad “antes/después”**

1. Genera 2–3 párrafos **antes** del ajuste y 2–3 **después**.
2. En parejas, pidan a los estudiantes resaltar cambios de **léxico**, **ritmo** y **giros**.
3. Cierre: ¿qué perdimos/ganamos con el *fine-tuning*? Relación con **atención** y **contexto**.

---

## 🔒 Licencias y procedencia de datos/modelo

* **Texto**: *Don Quijote* desde Project Gutenberg (*2000-0.txt*). Ver términos en la página del proyecto. ([Project Gutenberg][5])
* **Modelo base**: \[`datificate/gpt2-small-spanish`] (model card indica **Apache-2.0**). Revisa siempre el model card para uso y atribución. ([Hugging Face][6])
* **Código de este repo**: MIT (ajústalo si tu institución requiere otra licencia).

---

## 🆘 Solución de problemas

* **OOM / fuera de memoria**: baja `--train_bs` a `1` o `--block_size` a `64`; también puedes desactivar GPU y correr en CPU (más lento).
* **Texto repetitivo**: sube `--repetition_penalty` a `1.2–1.3`, o ajusta `--top_p` a `0.97`.
* **Genera menos de 5 oraciones**: aumenta `--max_new_tokens` (p. ej., 220) o baja `--temperature` (0.8).

---

## 📚 Lecturas recomendadas

* *Causal Language Modeling* (Transformers Docs). ([Hugging Face][7])
* Curso HF: *Training a causal language model from scratch* (cap. 7.6). ([Hugging Face][8])
* *FlashAttention* (para escalar eficientemente). ([arXiv][2])

---

## 📌 Cita sugerida (BibTeX)

```bibtex
@misc{quijote_minift_2025,
  title  = {Mini fine-tuning de GPT-2 (español) con ~100 líneas de Don Quijote},
  author = {Tu Nombre},
  year   = {2025},
  note   = {Versión educativa. Modelo base: datificate/gpt2-small-spanish; texto: Project Gutenberg.}
}
```

---

## ❤️ Créditos

* Hugging Face Transformers — guía de *language modeling*. ([Hugging Face][7])
* Project Gutenberg — *Don Quijote* (ed. UTF-8). ([Project Gutenberg][5])
* Modelo `datificate/gpt2-small-spanish` (ver model card/licencia). ([Hugging Face][6])

```
::contentReference[oaicite:13]{index=13}
```

[1]: https://huggingface.co/datificate/gpt2-small-spanish/tree/main?utm_source=chatgpt.com "datificate/gpt2-small-spanish at main"
[2]: https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[3]: https://ar5iv.labs.arxiv.org/html/2205.14135?utm_source=chatgpt.com "Fast and Memory-Efficient Exact Attention with IO-Awareness"
[4]: https://github.com/Dao-AILab/flash-attention?utm_source=chatgpt.com "Dao-AILab/flash-attention: Fast and memory-efficient ..."
[5]: https://www.gutenberg.org/files/2000/2000-0.txt?utm_source=chatgpt.com "El ingenioso hidalgo don Quijote de la Mancha"
[6]: https://huggingface.co/datificate/gpt2-small-spanish?utm_source=chatgpt.com "datificate/gpt2-small-spanish"
[7]: https://huggingface.co/docs/transformers/en/tasks/language_modeling?utm_source=chatgpt.com "Causal language modeling"
[8]: https://huggingface.co/learn/llm-course/en/chapter7/6?utm_source=chatgpt.com "Training a causal language model from scratch"
