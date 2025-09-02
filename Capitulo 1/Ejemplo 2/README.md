# LSTM Sentiment — IMDB (Keras)

Clasificador binario de sentimiento (positivo/negativo) con **Embedding → BiLSTM → Dense**, entrenado sobre **IMDB** (50k reseñas). El ejemplo es corto, 100% funcional, y corre sin cambios en **Google Colab** (CPU o GPU).

> El dataset IMDB de Keras ya viene vectorizado (listas de índices enteros) y con split train/test, ideal para demos rápidas. ([keras.io][1])

---

## 🔧 Qué contiene

* **Modelo:** `Embedding(V=20k) → Bidirectional(LSTM(128)) → Dense(64, ReLU) → Dense(1, sigmoid)`.
* **Entrenamiento estable:** `mask_zero=True`, `EarlyStopping` (monitor AUC), `gradient clipping`.
* **Inferencia libre:** función `predict_text()` que recibe texto crudo y devuelve probabilidad de “positivo”.

---

## 🚀 Ejecutar en Google Colab (copiar/pegar)

1. Abre [Google Colab](https://colab.research.google.com/), Crear una cuenta con usuario Google es gratis.
2. (Opcional) Activa GPU: **Runtime → Change runtime type → GPU**, luego **Save**. ([colab.research.google.com][2])
3. Crea una celda **nueva** de código danco click en (+Código) y **copia/pega** todo el siguiente bloque en **una sola celda** y ejecútala dando click en  el ícono de play en la parte superior izquierda de la celda:

NOTA: tambien puedes dar click en este link [documento compartido en google colab](https://colab.research.google.com/drive/1Yc61W46Z7LWzlNnF5_tz9SziLlnKlJeJ?usp=sharing)

> Referencias de API usadas: LSTM y Bidirectional en Keras/TensorFlow; `pad_sequences` y `EarlyStopping`. ([keras.io][3], [TensorFlow][4])

---

## 🧠 ¿Cómo interpretar la salida?

* El modelo termina en una **sigmoid** → devuelve un valor **p ∈ \[0, 1]**.
* **Umbral 0.5** (por defecto aquí):

  * `p ≥ 0.5` → **“positivo”**
  * `p < 0.5` → **“negativo”**
* Ejemplo:

  * `predict_text("...brilliant performances...") → (0.87, "positivo")`
    Significa que el modelo estima **87%** de probabilidad de que la reseña sea positiva.
* Sugerencias:

  * Para aplicaciones sensibles, calibra el umbral (p.ej., maximizando F1 o ajustándolo al coste de falsos positivos/negativos).
  * Reporta **AUC** y **accuracy** de test como guía de rendimiento global.

---

## 📦 Ejecutar localmente (opcional)

```bash
python -m venv .venv
source .venv/bin/activate    # en Windows: .venv\Scripts\activate
pip install -U "tensorflow>=2.13,<3.0"
python LSTM_imdb.py
```

> Si tu GPU es NVIDIA y tienes CUDA/cuDNN correctos, Keras usará kernels acelerados automáticamente cuando las condiciones lo permiten (p. ej., cuDNN LSTM). Revisa los docs de la capa LSTM para requisitos/argumentos compatibles. ([keras.io][3])

---

## 🧱 ¿Qué hace cada componente?

* **`Embedding(V, d, mask_zero=True)`**: convierte índices en vectores densos y **enmascara** el padding para que la LSTM lo ignore. ([TensorFlow][5])
* **`Bidirectional(LSTM(units))`**: procesa la secuencia izquierda→derecha y derecha→izquierda y concatena estados. Suele mejorar señales contextuales. ([keras.io][6])
* **`pad_sequences(...)`**: corta/rellena listas de índices para tener la misma longitud en batch. ([TensorFlow][4])
* **`EarlyStopping(...)`**: detiene el entrenamiento cuando la métrica monitorizada deja de mejorar (aquí, `val_auc`), restaurando los mejores pesos. ([keras.io][7])

---

## 📊 Resultados esperados

En 4–6 épocas, es común obtener **AUC ≈ 0.88–0.95** y **accuracy ≈ 0.83–0.90** (varía por semilla/hardware/versión). En GPU entrenará más rápido; en CPU también funciona (más lento, sobretodo el entrenamiento, puede tardar más de media hora).

---

## 🧪 Pruebas rápidas de inferencia

puedes crear una celda nueva de código: para esto en google colab da click en (+Código) y con la red ya entrenada (luego de los varios minutos que tomó entrenarla) puedes hacer nuevas inferencias:

Prueba frases propias (en inglés) en la celda final copia y pega este còdigo en una nueva celda:

```python
p, label = predict_text("I didn't like the movie at all, it was too long and predictable.")
print(p, label)
```

* Si `p=0.18 → "negativo"`.
* Si `p=0.76 → "positivo"`.

> Nota: El vocabulario está basado en frecuencia de IMDB; palabras fuera del top `VOCAB_SIZE` o desconocidas se marcan como `<OOV>`. Esto puede degradar algo la precisión en dominios muy distintos (e.g., noticias, ciencia).

---

## 🛠️ Solución de problemas

* **ImportError con `pad_sequences`:** usa `from tensorflow.keras.utils import pad_sequences` (o `from keras.utils import pad_sequences` según tu instalación). La ruta moderna en TF2 es `tf.keras.utils.pad_sequences`. ([TensorFlow][4])
* **No ves GPU en Colab:** ve a **Runtime → Change runtime type → GPU** y vuelve a ejecutar la celda inicial. ([colab.research.google.com][2])
* **Memoria insuficiente:** reduce `BATCH_SIZE` a 64 o `MAXLEN` a 150.

---

## 🔁 Reproducibilidad

* Fijamos semilla (`tf.random.set_seed`, `np.random.seed`).
* Anota versión de TensorFlow/Keras (se imprime al iniciar).
* Usa `EarlyStopping(restore_best_weights=True)` para reportar el mejor checkpoint, no el último.

---

## 📚 Referencias

* **Keras Datasets — IMDB** (detalles y loader). ([keras.io][1])
* **LSTM / Bidirectional (Keras API).** ([keras.io][3])
* **`pad_sequences` (TensorFlow API).** ([TensorFlow][4])
* **`EarlyStopping` (Keras callbacks).** ([keras.io][7])
* **Colab: cambiar runtime a GPU.** ([colab.research.google.com][2])

---

[1]: https://keras.io/api/datasets/imdb/?utm_source=chatgpt.com "IMDB movie review sentiment classification dataset"
[2]: https://colab.research.google.com/notebooks/pro.ipynb?utm_source=chatgpt.com "Making the Most of your Colab Subscription - Google"
[3]: https://keras.io/api/layers/recurrent_layers/lstm/?utm_source=chatgpt.com "LSTM layer"
[4]: https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences?utm_source=chatgpt.com "tf.keras.utils.pad_sequences | TensorFlow v2.16.1"
[5]: https://www.tensorflow.org/guide/keras/understanding_masking_and_padding?utm_source=chatgpt.com "Understanding masking & padding | TensorFlow Core"
[6]: https://keras.io/api/layers/recurrent_layers/bidirectional/?utm_source=chatgpt.com "Bidirectional layer"
[7]: https://keras.io/api/callbacks/early_stopping/?utm_source=chatgpt.com "EarlyStopping"
