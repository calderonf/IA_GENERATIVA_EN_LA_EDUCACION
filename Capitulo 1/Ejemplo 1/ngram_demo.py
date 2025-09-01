#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ngram_demo.py — Ejemplo autocontenido de modelos n‑grama en español.

Este script acompaña al Capítulo 1 del libro “IA generativa en la educación”
y permite entrenar, comparar y experimentar con modelos n‑grama de orden
1 a 3 (unigramas, bigramas y trigramas) sobre textos en español.  Incluye:

– Tokenización sencilla (oraciones y palabras) sin dependencias externas.
– Entrenamiento de modelos 1–3‑gramas por conteo.
– Tres técnicas de estimación:
  * **MLE** (máxima verosimilitud; sin suavizado; útil como contraste).
  * **add‑k** (Laplace/Lidstone), que suma un valor *k* a cada conteo.
  * **Kneser–Ney interpolado**, una versión docente con un descuento fijo *D*
    y un piso numérico que evita probabilidades nulas en corpora pequeños.
– Manejo de OOV: palabras con frecuencia ≤ `min_count` se mapean a `<unk>`.
– Utilidades didácticas: predicción de continuaciones (*top‑k*), generación de
  oraciones y evaluación por perplejidad con división 70/30 entre
  entrenamiento y prueba.

Notas:
– Filtramos `<s>`, `</s>` y `<unk>` del *top‑k* para que no “ensucien” la lista.
– El piso `eps` (muy pequeño) sólo se aplica si la probabilidad calculada
  queda en 0.
– En producción se utiliza la variante “Modified Kneser–Ney” (por ejemplo,
  mediante KenLM) que emplea descuentos distintos según el conteo.  Este
  script prioriza la claridad didáctica【6†L20-L60】.
"""

from __future__ import annotations
from collections import Counter, defaultdict
import math, random, re, argparse, textwrap
from typing import List, Tuple, Dict

# ---------------------------
# 1) Utilidades de texto
# ---------------------------

def sentence_tokenize(text: str) -> List[str]:
    """Divide en oraciones de manera sencilla sin dependencias externas."""
    sents = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def word_tokenize(sent: str) -> List[str]:
    """Tokenizador básico para español (minúsculas, letras/acentos, dígitos y apóstrofe)."""
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9']+", sent.lower())

def add_sentence_markers(sents: List[List[str]], order: int) -> List[List[str]]:
    """Añade <s> ... </s> para cada oración tokenizada, con (order-1) marcadores de inicio."""
    start = ['<s>'] * max(order - 1, 0)
    out = []
    for toks in sents:
        out.append(start + toks + ['</s>'])
    return out

# ---------------------------
# 2) Modelo N-grama
# ---------------------------

class NgramLM:
    """
    Modelo n‑grama con manejo de OOV y tres técnicas de suavizado.

    Este objeto implementa un modelo n‑grama de orden arbitrario (por
    defecto 3) para texto en español.  Permite entrenar el modelo
    mediante conteo directo y ofrece tres esquemas de estimación
    probabilística:

      * ``'mle'`` — máxima verosimilitud: se calcula la probabilidad
        condicional a partir de los conteos tal cual; cualquier n‑grama
        no observado recibe probabilidad cero.  Se usa principalmente
        como referencia para demostrar la necesidad de suavizado.

      * ``'add-k'`` — suavizado de Laplace/Lidstone: a cada conteo se
        suma una constante positiva ``k`` (por defecto 1.0), y el
        denominador se ajusta multiplicando ``k`` por el tamaño del
        vocabulario.  Con valores pequeños (p. ej. 0.1) es un
        suavizado razonable en corpora pequeños.

      * ``'kneser-ney'`` — Kneser–Ney interpolado: descuenta una
        cantidad fija ``discount`` (por defecto 0.75) de los conteos
        observados y redistribuye la probabilidad en función de cuántos
        contextos distintos preceden a cada palabra (probabilidad de
        continuación).  Esta versión interpolada se aplica a bigramas y
        trigramas; si el orden es 1, se recurre a un modelo unigrama
        simple.  Se incluye un pequeño piso numérico ``eps`` para
        evitar ceros en corpora diminutos.

    Además de las probabilidades, el modelo maneja vocabulario
    desconocido mediante el token ``<unk>``: cualquier palabra con
    frecuencia menor o igual a ``min_count`` en el entrenamiento se
    reemplaza por ``<unk>``.  Los marcadores ``<s>`` y ``</s>`` se
    añaden automáticamente al principio y al final de cada oración
    durante el entrenamiento.

    Los principales métodos de interés son:

    * :meth:`fit` — entrena el modelo a partir de una lista de textos.
    * :meth:`prob` — devuelve la probabilidad condicional de una
      palabra dada un contexto.
    * :meth:`topk_next` — lista las ``k`` palabras más probables que
      pueden seguir a un contexto.
    * :meth:`generate` — genera una oración muestreando según el modelo.
    * :meth:`perplexity` — calcula la perplejidad sobre un conjunto de
      textos de prueba (menor es mejor).
    """
    def __init__(self, order: int = 3, smoothing: str = 'kneser-ney',
                 k: float = 1.0, discount: float = 0.75,
                 min_count: int = 1, seed: int = 123):
        assert order >= 1
        assert smoothing in ('mle', 'add-k', 'kneser-ney')
        self.order = order
        self.smoothing = smoothing
        self.k = k
        self.discount = discount
        self.min_count = max(0, int(min_count))
        self.rng = random.Random(seed)

        # Piso numérico para evitar ceros puntuales en KN cuando el corpus es diminuto
        self.eps = 1e-8

        # Vocabulario y conteos
        self.vocab: set[str] = set()  # incluye <unk>, <s>, </s>
        self.ngram_counts: List[Counter] = [Counter() for _ in range(order)]   # n=1..order
        self.context_counts: List[Counter] = [Counter() for _ in range(order)] # contextos long. 0..order-1

        # Estadísticos para Kneser–Ney
        self.continuation_counts: Counter = Counter()  # N1+(. w): nº contextos distintos que preceden a w
        self.total_bigram_types: int = 0               # N1+(..): nº total de tipos de bigrama
        self.follow_types: Counter = Counter()         # N1+(h, .): nº de sucesores distintos tras el contexto h

        # Mapeo OOV (se rellena en fit)
        self._known_tokens: set[str] = set()

    # --------- Entrenamiento ---------

    def fit(self, texts: List[str]) -> None:
        """Entrena el modelo con una lista de documentos o párrafos en texto libre."""
        # 1) Tokeniza a nivel oración y palabra
        raw_sents = []
        for t in texts:
            for s in sentence_tokenize(t):
                raw_sents.append(word_tokenize(s))

        # 2) Construye vocab base y mapea OOV a <unk> (solo según TRAIN)
        uni = Counter(tok for s in raw_sents for tok in s)
        self._known_tokens = {w for w, c in uni.items() if c > self.min_count}
        self._known_tokens.add('<unk>')  # aseguramos que exista
        # 3) Aplica el mapeo a las oraciones
        mapped_sents = [[w if w in self._known_tokens else '<unk>' for w in s] for s in raw_sents]

        # 4) Añade marcadores y cuenta n-gramas
        marked = add_sentence_markers(mapped_sents, self.order)
        for toks in marked:
            self.vocab.update(toks)  # incluye <unk>, <s>, </s>
            for n in range(1, self.order + 1):
                for i in range(len(toks) - n + 1):
                    ngram = tuple(toks[i:i+n])
                    self.ngram_counts[n-1][ngram] += 1
                    context = ngram[:-1]
                    self.context_counts[n-1][context] += 1

        # 5) Precálculos para KN
        if self.order >= 2:
            bigram_counts = self.ngram_counts[1]  # (w_{i-1}, w_i)
            contexts_seen: Dict[str, set] = defaultdict(set)
            for (w_prev, w), _c in bigram_counts.items():
                contexts_seen[w].add(w_prev)
                self.follow_types[(w_prev,)] += 1
            self.continuation_counts = Counter({w: len(ctxs) for w, ctxs in contexts_seen.items()})
            self.total_bigram_types = len(bigram_counts)

        if self.order >= 3:
            trigram_counts = self.ngram_counts[2]
            for (w1, w2, _w3), _c in trigram_counts.items():
                self.follow_types[(w1, w2)] += 1

    # --------- Probabilidades básicas ---------

    def _map_oov(self, w: str) -> str:
        """Mapea tokens desconocidos a <unk> según el vocab construido en TRAIN."""
        return w if w in self.vocab else '<unk>'

    def _prob_mle(self, context: Tuple[str, ...], w: str) -> float:
        n = len(context) + 1
        w = self._map_oov(w)
        c_ng = self.ngram_counts[n-1][context + (w,)]
        c_ctx = self.context_counts[n-1][context]
        if c_ctx == 0:
            total = sum(self.ngram_counts[0].values())
            return self.ngram_counts[0][(w,)] / total if total > 0 else 0.0
        return c_ng / c_ctx

    def _prob_addk(self, context: Tuple[str, ...], w: str) -> float:
        n = len(context) + 1
        w = self._map_oov(w)
        V = len(self.vocab)
        c_ng = self.ngram_counts[n-1][context + (w,)]
        c_ctx = self.context_counts[n-1][context]
        return (c_ng + self.k) / (c_ctx + self.k * V)

    # --------- Kneser–Ney (interpolado) ---------

    def _prob_kn_bigram(self, w_prev: str, w: str) -> float:
        D = self.discount
        w = self._map_oov(w)
        w_prev = self._map_oov(w_prev)

        c_big = self.ngram_counts[1][(w_prev, w)]
        c_prev = self.context_counts[1][(w_prev,)]
        # P_cont(w) = N1+(. w) / N1+(..)
        N1_cont_w = self.continuation_counts.get(w, 0)
        P_cont = (N1_cont_w / self.total_bigram_types) if self.total_bigram_types > 0 else 0.0

        if c_prev == 0:
            p = P_cont
        else:
            N1_follow = self.follow_types[(w_prev,)]
            lam = D * N1_follow / c_prev
            p = max(c_big - D, 0) / c_prev + lam * P_cont

        if p == 0.0:
            V = max(len(self.vocab), 1)
            p = self.eps / V
        return p

    def _prob_kn(self, context: Tuple[str, ...], w: str) -> float:
        w = self._map_oov(w)
        context = tuple(self._map_oov(tok) for tok in context)

        # Unigrama “KN”: prob. de continuación si hay bigramas; si no, uniforme
        if len(context) == 0:
            N1_cont_w = self.continuation_counts.get(w, 0)
            if self.total_bigram_types > 0:
                p = N1_cont_w / self.total_bigram_types
            else:
                p = 1.0 / max(len(self.vocab), 1)
            if p == 0.0:
                p = self.eps / max(len(self.vocab), 1)
            return p

        if len(context) == 1:
            return self._prob_kn_bigram(context[0], w)

        # Trigrama KN interpolado con backoff recursivo
        D = self.discount
        c_tri = self.ngram_counts[2][(context[0], context[1], w)]
        c_ctx = self.context_counts[2][(context[0], context[1])]
        if c_ctx == 0:
            p = self._prob_kn(context[1:], w)
        else:
            N1_follow = self.follow_types[(context[0], context[1])]
            lam = D * N1_follow / c_ctx
            p = max(c_tri - D, 0) / c_ctx + lam * self._prob_kn(context[1:], w)

        if p == 0.0:
            V = max(len(self.vocab), 1)
            p = self.eps / V
        return p

    # --------- Interfaz pública ---------

    def prob(self, context: List[str] | Tuple[str, ...], w: str) -> float:
        """Probabilidad de la siguiente palabra w dado el contexto (truncado a order-1)."""
        context = tuple(context[-(self.order - 1):]) if self.order > 1 else tuple()
        if self.smoothing == 'mle':
            return self._prob_mle(context, w)
        if self.smoothing == 'add-k':
            return self._prob_addk(context, w)
        if self.smoothing == 'kneser-ney':
            if self.order == 1:
                # fallback unigrama (no KN real)
                w = self._map_oov(w)
                total = sum(self.ngram_counts[0].values())
                p = self.ngram_counts[0][(w,)] / total if total > 0 else 0.0
                if p == 0.0:
                    p = self.eps / max(len(self.vocab), 1)
                return p
            return self._prob_kn(context, w)
        raise ValueError("Suavizado desconocido")

    def topk_next(self, context: List[str], k: int = 10) -> List[Tuple[str, float]]:
        """Top-k continuaciones probables (excluye <s>, </s> y <unk>)."""
        context = tuple(context[-(self.order - 1):]) if self.order > 1 else tuple()
        banned = {'<s>', '</s>', '<unk>'}
        candidates = [w for w in self.vocab if w not in banned]
        scores = [(w, self.prob(context, w)) for w in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def generate(self, max_tokens: int = 30) -> str:
        """Genera una oración (detiene en </s>)."""
        context = ['<s>'] * (self.order - 1)
        out = []
        for _ in range(max_tokens):
            w = self._sample_next(context)
            if w == '</s>':
                break
            if w == '<unk>':  # evita imprimir <unk> en texto generado
                continue
            out.append(w)
            context.append(w)
        return ' '.join(out)

    def _sample_next(self, context: List[str]) -> str:
        context = tuple(context[-(self.order - 1):]) if self.order > 1 else tuple()
        candidates = [w for w in self.vocab if w != '<s>']  # permitimos </s> para terminar
        probs = [self.prob(context, w) for w in candidates]
        s = sum(probs)
        if s <= 0.0:
            probs = [1.0 / len(candidates)] * len(candidates)
        else:
            probs = [p / s for p in probs]  # normalizamos para muestreo
        r = self.rng.random()
        acc = 0.0
        for w, p in zip(candidates, probs):
            acc += p
            if r <= acc:
                return w
        return candidates[-1]

    def log_prob_sentence(self, tokens: List[str]) -> float:
        """Log-probabilidad base 2 de una oración (sin <s>; añadimos </s>)."""
        context = ['<s>'] * (self.order - 1)
        logp = 0.0
        for w in tokens + ['</s>']:
            w_m = self._map_oov(w)
            p = self.prob(context, w_m)
            if p <= 0.0:
                return float('-inf')
            logp += math.log(p, 2)
            context.append(w_m)
        return logp

    def perplexity(self, texts: List[str]) -> float:
        """Perplejidad sobre una lista de textos (menor = mejor)."""
        sents = []
        for t in texts:
            for s in sentence_tokenize(t):
                sents.append([self._map_oov(w) for w in word_tokenize(s)])
        marked = add_sentence_markers(sents, self.order)

        total_tokens = 0
        sum_logp = 0.0
        for toks in marked:
            toks_wo_markers = [tok for tok in toks if tok not in ('<s>', '</s>')]
            total_tokens += len(toks_wo_markers) + 1  # contamos </s>
            lp = self.log_prob_sentence(toks_wo_markers)
            if lp == float('-inf'):
                return float('inf')
            sum_logp += lp

        avg_logp = sum_logp / max(total_tokens, 1)
        return 2 ** (-avg_logp)

# ---------------------------
# 3) Demo y utilidad CLI
# ---------------------------

DEMO_CORPUS = """
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
"""

def demo():
    # Split simple train/test
    sents = sentence_tokenize(DEMO_CORPUS)
    cutoff = int(0.7 * len(sents)) if len(sents) > 1 else 1
    train_text = " ".join(sents[:cutoff]) or DEMO_CORPUS
    test_text  = " ".join(sents[cutoff:]) or DEMO_CORPUS

    print("\n=== Entrenamiento (trigrama, Kneser–Ney D=0.75, min_count=1) ===")
    # En el ejemplo de demo conservamos min_count=1; con un corpus pequeño esto
    # colapsa parte del vocabulario, lo cual se ilustra en el README. Para
    # obtener un vocabulario más amplio puedes ejecutar el script con
    # --min_count 0.
    lm_kn = NgramLM(order=3, smoothing='kneser-ney', discount=0.75, min_count=1, seed=7)
    lm_kn.fit([train_text])
    print(f"Vocabulario: {len(lm_kn.vocab)} palabras (incluye <unk>, <s>, </s>)")
    print("Top-10 continuaciones de contexto ['la'] (sin </s> ni <unk>):")
    for w,p in lm_kn.topk_next(['la'], k=10):
        print(f"  {w:15s}  {p:.4f}")

    print("\nGeneraciones:")
    for _ in range(3):
        print("  •", lm_kn.generate(max_tokens=20))

    pp_kn = lm_kn.perplexity([test_text])
    print(f"\nPerplejidad (test) KN trigram: {pp_kn:.3f}")

    print("\n=== Comparativa de suavizados (trigrama) ===")
    for name, sm, extra in [
        ('MLE', 'mle', {}),
        ('add-k=0.1', 'add-k', {'k': 0.1}),
        ('Kneser-Ney', 'kneser-ney', {'discount': 0.75})
    ]:
        lm = NgramLM(order=3, smoothing=sm, min_count=1, seed=7, **extra)
        lm.fit([train_text])
        pp = lm.perplexity([test_text])
        print(f"  {name:12s} -> Perplejidad: {pp if pp != float('inf') else 'inf'}")

def main():
    parser = argparse.ArgumentParser(
        description="Demostración autocontenida de modelos n-grama (con OOV y Kneser–Ney).")
    parser.add_argument("--file", type=str, default=None,
                        help="Ruta a un archivo de texto propio (UTF-8). Si no se pasa, usa el mini-corpus de ejemplo.")
    parser.add_argument("--order", type=int, default=3, help="Orden del modelo (1, 2 o 3).")
    parser.add_argument("--smoothing", type=str, default="kneser-ney",
                        choices=["mle","add-k","kneser-ney"], help="Técnica de estimación.")
    parser.add_argument("--k", type=float, default=0.1, help="Parámetro k para add-k.")
    parser.add_argument("--discount", type=float, default=0.75, help="Descuento D para Kneser–Ney.")
    parser.add_argument("--min_count", type=int, default=1,
                        help="Umbral de frecuencia para mapear a <unk> (OOV). Palabras con frecuencia \
                        menor o igual a este valor se reemplazan por <unk>. Fija 0 para conservar todas \
                        las palabras vistas en entrenamiento.")
    parser.add_argument("--gen", type=int, default=0, help="Número de oraciones a generar tras entrenar.")
    parser.add_argument("--topk", type=str, default="", help="Contexto para top-10 continuaciones (p.ej.: \"el docente\").")
    args = parser.parse_args()

    # Carga de datos
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = DEMO_CORPUS

    # División sencilla 70/30
    sents = sentence_tokenize(raw)
    cutoff = max(1, int(0.7 * len(sents)))
    train_text = " ".join(sents[:cutoff]) or raw
    test_text  = " ".join(sents[cutoff:]) or raw

    # Entrenar
    lm = NgramLM(order=args.order, smoothing=args.smoothing,
                 k=args.k, discount=args.discount, min_count=args.min_count, seed=7)
    lm.fit([train_text])

    print(textwrap.dedent(f"""
    --- Resumen del modelo ---
    Orden:            {args.order}
    Suavizado:        {args.smoothing} (k={args.k if args.smoothing=='add-k' else '-'}, D={args.discount if args.smoothing=='kneser-ney' else '-'})
    min_count:        {args.min_count}  (palabras con freq <= min_count -> <unk>)
    Vocabulario:      {len(lm.vocab)} palabras (incluye <unk>, <s>, </s>)
    Oraciones train:  {len(sentence_tokenize(train_text))}
    Oraciones test:   {len(sentence_tokenize(test_text))}
    """).strip())

    # Top-k continuaciones
    if args.topk:
        ctx = word_tokenize(args.topk)
        print(f"\nTop-10 continuaciones para contexto {ctx} (sin </s> ni <unk>):")
        for w,p in lm.topk_next(ctx, k=10):
            print(f"  {w:15s}  {p:.4f}")

    # Generación
    if args.gen > 0:
        print("\nMuestras generadas:")
        for _ in range(args.gen):
            print("  •", lm.generate(max_tokens=20))

    # Perplejidad
    pp = lm.perplexity([test_text])
    print(f"\nPerplejidad en test: {pp if pp != float('inf') else 'inf'}\n")
    print("Sugerencia: prueba --order 2 o --smoothing add-k para corpus muy pequeños; ajusta"
          " --min_count 0 si ves que el vocabulario colapsa. Para proyectos grandes,"
          " considera KenLM (Modified Kneser–Ney).")

if __name__ == "__main__":
    # Si llamas al script sin argumentos, corre una demo ilustrativa.
    import sys
    if len(sys.argv) == 1:
        demo()
    else:
        main()
