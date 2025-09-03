
# Mini fine-tuning de GPT-2 (español) con ~100 líneas de *Don Quijote*

> Ajuste educativo **ultra-ligero** de un GPT-2 pequeño en español para **completar un cuento de 5 oraciones**.  
> Diseñado para **Google Colab** (GPU recomendada) o ejecución local.

---

## 🧭 Qué hace este repo

1. Descarga **100 líneas** de *Don Quijote* (dominio público) desde Project Gutenberg. 
2. Prepara un **dataset mínimo** (solo para demostración didáctica).  
3. Ajusta el modelo **`datificate/gpt2-small-spanish`** (licencia Apache-2.0 según su model card). 
4. Genera un **cuento de 5 oraciones** con sampling controlado.  
5. Guarda el modelo ajustado en `gpt2_es_quijote_minift/final/`.

> Para los curiosos: el procedimiento sigue la **receta oficial** de *Causal Language Modeling* de Transformers (sección “Language modeling”).

---

## ⚡️ Ejecución en Google Colab (recomendada)

1. **Abrir Colab** y activar GPU: `Entorno de ejecución → Cambiar tipo de entorno → GPU`.
2. Crear una celda e instalar dependencias:
    ```bash
    pip install "transformers==4.45.2" "datasets==2.20.0" "accelerate==0.33.0" sentencepiece
    ```

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

## ❤️ Créditos

* Hugging Face Transformers — guía de *language modeling*. ([Hugging Face][7])
* Project Gutenberg — *Don Quijote* (ed. UTF-8). ([Project Gutenberg][5])
* Modelo `datificate/gpt2-small-spanish` (ver model card/licencia). ([Hugging Face][6])



[1]: https://huggingface.co/datificate/gpt2-small-spanish/tree/main "datificate/gpt2-small-spanish at main"
[2]: https://arxiv.org/abs/2205.14135 "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
[3]: https://ar5iv.labs.arxiv.org/html/2205.14135 "Fast and Memory-Efficient Exact Attention with IO-Awareness"
[4]: https://github.com/Dao-AILab/flash-attention "Dao-AILab/flash-attention: Fast and memory-efficient ..."
[5]: https://www.gutenberg.org/files/2000/2000-0.txt "El ingenioso hidalgo don Quijote de la Mancha"
[6]: https://huggingface.co/datificate/gpt2-small-spanish "datificate/gpt2-small-spanish"
[7]: https://huggingface.co/docs/transformers/en/tasks/language_modeling "Causal language modeling"
[8]: https://huggingface.co/learn/llm-course/en/chapter7/6 "Training a causal language model from scratch"
