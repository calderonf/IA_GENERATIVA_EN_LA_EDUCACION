
# Mini fine-tuning de GPT-2 (español) con ~100 líneas de *Don Quijote*

> Ajuste educativo **ultra-ligero** de un GPT-2 pequeño en español para **completar un prompt corto**.  
> Diseñado para **Google Colab** (GPU recomendada) o ejecución local.

---

## 🧭 Qué hace este repo

1. Descarga **algunas líneas** de *Don Quijote* (dominio público) desde Project Gutenberg. 
2. Prepara un **dataset mínimo** (solo para demostración didáctica).  
3. Ajusta el modelo **`datificate/gpt2-small-spanish`** (licencia Apache-2.0 según su model card). 
4. Genera un **cuento de 5 oraciones** con muestreo controlado.  
5. Guarda el modelo ajustado en `gpt2_es_quijote_minift/final/`.

> Tambien puedes encontrar este ejemplo directamente en: [google colab](https://colab.research.google.com/drive/1uSRsAuAs2PqXKouHunf8URDaG1rNv-Ut?usp=sharing)

> Para los curiosos: el procedimiento sigue la **receta oficial** de *Causal Language Modeling* de Transformers (sección “Language modeling”), ver al final en créditos.

---

## ⚡️ Ejecución en Google Colab (recomendada)

1. **Abrir Colab** y activar GPU: `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4` o alguna otra que tenga disponible que tenga las letras GPU NO CPU o TPU, y dar click en `Guargar` .
2. Crear una celda `+Código` y copiar y pegar el código de miniift.py en esta celda
3. Ejecuta: dando click en `Ejecutar todas` o en el simbolo de play que sale al acercar el mouse a la parte superior izquierda de la celda que quieres ejecutar
4. Puedes crear otra celda y ejecutar:
    ```python
    gen = load_generator_from_disk()
    generate_text("Escribe cinco oraciones sobre un colegio de monjas quijotescas.", generator=gen)#acá el prompt entre ""
    ```

**Salida esperada (resumen):**

* Métrica `eval_loss` y su **perplejidad aproximada**.
* Un **cuento de 5 oraciones** con rasgos estilísticos del corpus breve (puede sonar arcaizante).

---


## 🔧 Parámetros útiles (CLI)
En la parte superior del script puedes ajustar directamente las variables.

| Variable             | Default     | Explicación “amigable”                                           |
| -------------------- | ----------- | ---------------------------------------------------------------- |
| `EPOCHS`             | `2`         | **Vueltas al cuaderno**; con pocos datos, 1–3 bastan.            |
| `LR`                 | `1e-4`      | **Qué tan rápido corrijo**; grande = inestable, pequeño = lento. |
| `BLOCK_SIZE`         | `128`       | **Ventana de contexto**; más grande consume más VRAM.            |
| `TRAIN_BS`           | `2`         | Tamaño de batch (train).                                         |
| `GRAD_ACCUM`         | `4`         | **Acumulación de gradientes** (simula batches mayores).          |
| `MAX_NEW_TOKENS`     | `160`       | Largo máximo del texto **nuevo**.                                |
| `TEMPERATURE`        | `0.9`       | Creatividad: ↑ temperatura ⇒ ↑ variedad.                         |
| `TOP_K` / `TOP_P`    | `50`/`0.95` | Sampling controlado: **suerte con cinturón de seguridad**.       |
| `REPETITION_PENALTY` | `1.1`       | Penaliza repeticiones molestas.                                  |

---

## 🧪 ¿Qué aprendizaje ofrece?

* **Transferencia de estilo** con **muy pocos datos**: verás vocabulario y cadencia “quijotesca”.
* **Limitaciones**: el dataset diminuto propicia **sobreajuste**; no es un experimento de producción.
* **Buena práctica**: compara generaciones **antes** y **después** del ajuste y pide a tus estudiantes identificar “huellas” del corpus con el que se está entrenando aumenta N_LINES = 500  al inicio para entrenar con algo más del corpus, que pasa?

Para extender:

* Mezcla textos contemporáneos para un tono más moderno.
* Incrementa `EPOCHS` y reduce `LR` para un ajuste más fino (si tienes más datos).
* Explora kernels/optimizaciones modernas si escalas (p. ej., *FlashAttention*). ([arXiv][2], [ar5iv][3], [GitHub][4]) (avanzado)

---

## 🧰 Estructura de archivos

```
.
├── gpt2_es_minift.py          # Script principal (entrena, evalúa, genera y guarda)
├── don_quijote_100.txt        # Se crea automáticamente
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
* **Código de este repo**: MIT .

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


## Enlaces y referencias rápidas

1. [datificate/gpt2-small-spanish – árbol completo](https://huggingface.co/datificate/gpt2-small-spanish/tree/main) 
2. [FlashAttention (paper en arXiv)](https://arxiv.org/abs/2205.14135) 
3. [FlashAttention (versión web ar5iv)](https://ar5iv.labs.arxiv.org/html/2205.14135)
4. [Repositorio oficial de FlashAttention](https://github.com/Dao-AILab/flash-attention)
5. [Don Quijote de la Mancha – Proyecto Gutenberg](https://www.gutenberg.org/files/2000/2000-0.txt)
6. [Modelo GPT-2 small en español – Hugging Face](https://huggingface.co/datificate/gpt2-small-spanish)
7. [Guía oficial: Causal Language Modeling](https://huggingface.co/docs/transformers/en/tasks/language_modeling)
8. [Curso LLM: entrenar un modelo causal desde cero](https://huggingface.co/learn/llm-course/en/chapter7/6) 
