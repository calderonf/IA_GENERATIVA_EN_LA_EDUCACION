# LSTM para análisis de sentimientos en IMDB (binario)
# Francisco-ready: corto, claro y corre "tal cual" en CPU o GPU.
# lo ideal es ejecutarlo en google colab, ver el README.md para obtener
# las instrucciones en detalle 

import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------
# 1) Configuración reproducible
# ---------------------------
SEED = 13
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ---------------------------
# 2) Hiperparámetros
# ---------------------------
VOCAB_SIZE = 20000    # top palabras por frecuencia
MAXLEN     = 200      # longitud fija de secuencia
EMB_DIM    = 128
UNITS      = 128
BATCH_SIZE = 128
EPOCHS     = 6

# ---------------------------
# 3) Carga de datos (IMDB Keras)
#    Devuelve índices (enteros) ya mapeados por frecuencia
# ---------------------------
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
# Pad/truncate a longitud fija
x_train = pad_sequences(x_train, maxlen=MAXLEN)
x_test  = pad_sequences(x_test,  maxlen=MAXLEN)

# ---------------------------
# 4) Modelo: Embedding -> BiLSTM -> Dense(sigmoid)
#    mask_zero=True ignora los ceros del padding
# ---------------------------
model = models.Sequential([
    layers.Embedding(VOCAB_SIZE, EMB_DIM, input_length=MAXLEN, mask_zero=True),
    layers.Bidirectional(layers.LSTM(UNITS, dropout=0.2, recurrent_dropout=0.2)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ---------------------------
# 5) Entrenamiento con EarlyStopping
# ---------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=2, restore_best_weights=True
    )
]

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

# ---------------------------
# 6) Evaluación en test
# ---------------------------
test_loss, test_acc, test_auc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test  - Loss: {test_loss:.4f}  Acc: {test_acc:.3f}  AUC: {test_auc:.3f}")

# ---------------------------
# 7) Inferencia en texto libre
#    Usamos el word_index oficial de IMDB para convertir texto a índices.
#    OJO: Keras reserva 0 (padding), 1 (start), 2 (OOV). index_from=3.
# ---------------------------
word_index = imdb.get_word_index()
index_from = 3  # Keras reserva 0,1,2

def encode_text(s, word_index, num_words=VOCAB_SIZE, index_from=3):
    # tokenización súper simple (educativa)
    tokens = re.findall(r"[A-Za-z0-9']+", s.lower())
    seq = [1]  # start token (por convención Keras IMDB)
    for w in tokens:
        idx = word_index.get(w, None)
        if idx is not None:
            idx = idx + index_from
            if idx < num_words:
                seq.append(idx)
            else:
                seq.append(2)  # OOV si está fuera del vocab recortado
        else:
            seq.append(2)      # OOV si palabra desconocida
    return pad_sequences([seq], maxlen=MAXLEN)

def predict_text(s):
    x = encode_text(s, word_index, num_words=VOCAB_SIZE, index_from=index_from)
    p = float(model.predict(x, verbose=0)[0][0])
    label = "positivo" if p >= 0.5 else "negativo"
    return p, label

ejemplo_pos = "An outstanding movie with brilliant performances and a touching story."
ejemplo_neg = "Terribly boring and a complete waste of time."

print("Ejemplo +:", predict_text(ejemplo_pos))
print("Ejemplo -:", predict_text(ejemplo_neg))
