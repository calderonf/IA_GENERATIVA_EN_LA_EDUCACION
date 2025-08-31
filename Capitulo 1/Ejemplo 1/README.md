
# Cómo usarlo

* **Ejecutar la demo** (usa un mini-corpus interno, compara suavizados y genera oraciones):

```bash
python ngram_demo.py
```

* **Usar tu propio texto** (por ejemplo, `mis_clases.txt`):

```bash
python ngram_demo.py --file mis_clases.txt --order 3 --smoothing kneser-ney --gen 3 --topk "el docente"
```

* **Probar suavizados**:

```bash
# MLE (sin suavizado) — solo para ver por qué no es buena idea
python ngram_demo.py --file mis_clases.txt --smoothing mle

# add-k con k=0.1
python ngram_demo.py --file mis_clases.txt --smoothing add-k --k 0.1

# Kneser–Ney (D=0.75 por defecto)
python ngram_demo.py --file mis_clases.txt --smoothing kneser-ney
```

**Qué mirar:**

* *Top-k continuaciones* ayudan a ver el “estilo” aprendido.
* *Perplejidad* (menor = mejor) te da una cifra objetiva para comparar configuraciones.
* *Generaciones* muestran la fluidez local, pero recuerda que n-gramas **no** garantizan coherencia de largo alcance (de ahí la migración histórica hacia RNN y, luego, *transformers*). ([Stanford University][1])

al ejecutar el ejemplo deberias ver algo como:


```bash
=== Entrenamiento (trigrama, Kneser–Ney D=0.75, min_count=1) ===
Vocabulario: 5 palabras (incluye <unk>, <s>, </s>)
Top-10 continuaciones de contexto ['el'] (sin </s> ni <unk>):
  la               0.0417
  el               0.0417

Generaciones:
  • el
  • 
  • la la la

Perplejidad (test) KN trigram: 2.370

=== Comparativa de suavizados (trigrama) ===
  MLE          -> Perplejidad: inf
  add-k=0.1    -> Perplejidad: 2.283475729677735
  Kneser-Ney   -> Perplejidad: 2.3703655852843495
```

---

## Notas y referencias prácticas

* **¿Por qué Kneser–Ney?** Es el suavizado que históricamente obtiene mejores resultados en *benchmarks* clásicos; la variante **modificada** (descuentos distintos según el conteo) es la que implementan *toolkits* como **KenLM**. Este script implementa la versión **interpolada** con un único descuento $D=0.75$, suficiente para docencia y corpus pequeños; para proyectos serios en producción, usa KenLM o similares. ([u.cs.biu.ac.il][2], [Stanford University][5], [GitHub][3], [kheafield.com][4])
* **Capítulo de referencia** (explicación clara de n-gramas, perplejidad y suavizados, con ejemplos): *SLP3* de Jurafsky & Martin. ([Stanford University][1])
* **Lecturas ligeras sobre KN** (intuición): entradas de blog con derivaciones y ejemplos. ([foldl][6], [Smitha Milli][7])

---


Te explico por qué y cómo mejorarlo para que el demo sea más “interesante”.

## Cómo leer tu salida

* **Vocabulario: 5 palabras**
  Con `min_count=1`, todo token que aparece **1 vez o menos** se mapea a `<unk>`. En un corpus tan pequeño, eso colapsa casi todo y te quedan básicamente: `{'<s>', '</s>', '<unk>', 'el', 'la'}`. De ahí que el top-10 para `['el']` solo muestre `el` y `la`.

* **Generaciones muy cortas y repetitivas**
  Como el vocabulario “útil” quedó en 2 palabras, el modelo apenas puede combinar nada. Es normal que salgan repeticiones.

* **Perplejidad**

  * **MLE** → `inf` (esperable): cualquier n-grama no visto en train recibe probabilidad 0.
  * **add-k=0.1** (2.283) y **KN** (2.370): valores **pequeños** porque el modelo, con dos opciones reales, tiene muy poca incertidumbre. Que add-k gane a KN aquí no es raro con corpora diminutos; KN brilla con más datos.

## Para obtener resultados más útiles

1. **No colapses el vocabulario**
   Ejecuta con `min_count=0` para **conservar todas** las palabras vistas en train:

   ```bash
   python ngram_demo.py --smoothing kneser-ney --order 3 --discount 0.75 --min_count 0 --gen 3 --topk "el"
   ```

   Verás un vocabulario mayor, top-k más variado y generaciones menos repetitivas.

2. **Si el corpus sigue siendo pequeño, prueba bigramas**
   Los trigramas son muy “hambrientos” de datos:

   ```bash
   python ngram_demo.py --smoothing add-k --k 0.1 --order 2 --min_count 0 --gen 3 --topk "el"
   ```

   En minidatasets, **bigramas + add-k** suelen comportarse mejor y dar perplejidad más estable.

3. **Aumenta un poco el corpus (aunque sea 10–20 frases)**
   Cuantas más oraciones, más n-gramas no colapsan. Puedes crear un archivo, por ejemplo `corpus_edu.txt`, con algo tipo:

   ```
   El docente explicó el concepto con ejemplos de aula.
   La estudiante comparó dos métodos y justificó su elección.
   El grupo debatió la validez de las fuentes consultadas.
   La profesora pidió evidencias y referencias actualizadas.
   El equipo documentó sus decisiones y reflexionó sobre el proceso.
   La rúbrica evaluó claridad, precisión y uso ético de la IA.
   El docente revisó los borradores y sugirió mejoras puntuales.
   La clase analizó sesgos y limitaciones de los modelos generativos.
   El estudiante citó correctamente el material utilizado.
   La actividad integró lectura crítica y producción colaborativa.
   En el laboratorio, la guía describió los pasos de la práctica.
   La retroalimentación incluyó ejemplos concretos y enlaces.
   El proyecto final articuló objetivos, método y resultados.
   La revisión por pares ayudó a detectar errores frecuentes.
   El docente cerró con recomendaciones para el siguiente módulo.
   ```

   Luego:

   ```bash
   python ngram_demo.py --file corpus_edu.txt --order 3 --smoothing kneser-ney --discount 0.75 --min_count 0 --gen 5 --topk "la"
   ```

4. **Ajustes finos si quieres experimentar**

   * KN: prueba `--discount 0.5` a `1.0`.
   * add-k: prueba `--k 0.1` a `1.0`.
   * Observa perplejidad y calidad de generaciones.

5. **Tip de inspección rápida**
   Si quieres ver qué quedó realmente en el vocabulario, tras entrenar (línea donde se imprime el vocab), añade:

   ```python
   print("Ejemplo de vocab:", sorted(list(lm_kn.vocab))[:30])
   ```

   Así confirmas si `min_count` te está dejando fuera palabras.

---

### Conclusión breve

* Lo que viste es **consistente**: el colapso del vocabulario por `min_count=1` deja casi solo `el/la`, y por eso la perplejidad es baja y las generaciones se repiten.
* Para un demo más expresivo: **usa `min_count=0` y/o más frases**, y si el corpus es chico, empieza con **bigramas + add-k**. Con un poco más de texto, **Kneser–Ney** empieza a superar a add-k y verás top-k y generaciones mucho más naturales.
