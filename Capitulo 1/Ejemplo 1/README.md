
# Ejemplo de Modelos N-grama en Español

Este repositorio contiene un ejemplo autocontenido en Python para demostrar el funcionamiento de **modelos de lenguaje basados en N-gramas**, como se describe en el **Capítulo 1 del libro “IA Generativa en la Educación”**.

El objetivo es ilustrar cómo, a partir de un corpus de texto, se pueden construir distribuciones de probabilidad para predecir la siguiente palabra, mostrando el paso previo a los modelos modernos como *transformers*.

---

## 📂 Archivos principales

- `ngram_demo.py`: Script en Python que construye y utiliza un modelo N-grama simple.
- `README.md`: Este archivo con instrucciones y contexto.

---

## ▶️ Ejecución rápida

### 1. Requisitos

Este ejemplo está desarrollado en **Python 3.8+** y no requiere librerías externas más allá de la estándar.

Opcionalmente:

Puedes copiar y pegar el código en un  [colab](https://colab.research.google.com/) para ejecutarlo rápidamente.

Puedes instalar [matplotlib](https://matplotlib.org/) si deseas graficar tu mismo las distribuciones:

```bash
pip install matplotlib
````

### 2. Uso básico

**Ver ayuda** y opciones disponibles:

```bash
python ngram_demo.py --help
```
**Ejecutar la demo** (usa un mini-corpus interno, compara suavizados y genera oraciones):

```bash
python ngram_demo.py
```

**Ejecuta el script con bigramas y 30 tokens generados usando archivo de ejemplo corpus.txt**:

```powershell
python .\ngram_demo.py --file .\corpus.txt --order 2 --smoothing kneser-ney --gen 30
```

#### Variantes útiles

* Bigramas con suavizado Add-k:

```powershell
python .\ngram_demo.py --file .\corpus.txt --order 2 --smoothing add-k --k 0.1 --gen 30
```

* Trigramas (puede sonar más “corpus-dependiente” con textos cortos):

```powershell
python .\ngram_demo.py --file .\corpus.txt --order 3 --smoothing kneser-ney --gen 30
```

* Mostrar top-k continuaciones del contexto por defecto (además de generación):

```powershell
python .\ngram_demo.py --file .\corpus.txt --order 2 --smoothing kneser-ney --topk 10 --gen 30
```

#### Sin `corpus.txt`

Si quieres “probar rápido” sin archivo, el script trae un **corpus de demo** y funciona ejecutando simplemente:

```powershell
python .\ngram_demo.py
```

(o con parámetros, por ejemplo:)

```powershell
python .\ngram_demo.py --order 3 --smoothing kneser-ney --gen 30
```



---


## 📖 Relación con el libro

Este ejemplo está vinculado al **Capítulo 1: Historia y evolución de la IA generativa**, en la subsección dedicada a los modelos estadísticos de lenguaje.

Incluye el código QR que enlaza directamente a este repositorio:

![QR](qrcode_github_ejemplo1.png)

Para más contexto, consulta el texto guia.

---

## 🔗 Enlaces útiles

* Repositorio en GitHub: [IA\_GENERATIVA\_EN\_LA\_EDUCACION](https://github.com/calderonf/IA_GENERATIVA_EN_LA_EDUCACION)
* Documentación de Python: [https://docs.python.org/3/](https://docs.python.org/3/)

---

## ✨ Créditos

Autor: **Francisco Carlos Calderón Bocanegra**
Pontificia Universidad Javeriana – Departamento de Electrónica

Este ejemplo es de libre uso con fines educativos.

```