# Recuperación activa (*active recall*)

Prompts del **capítulo 6**, sección «Técnicas cognitivas potenciadas por IA».

Recuperar información desde la memoria, sin mirar los apuntes. Es la técnica de estudio con mejor respaldo empírico: en la revisión de diez técnicas de uso común de Dunlosky et al. (2013), la práctica de recuperación es una de las dos únicas que alcanzan la calificación de **utilidad alta**. Subrayar y releer, las dos más usadas por los estudiantes, quedan en el grupo de utilidad baja.

Releer apuntes se siente productivo y no lo es. Intentar responder sin mirarlos sí lo es, aunque se sienta peor.

---

## 1. Generación de *flashcards* por nivel

```text
Crea 20 tarjetas de estudio sobre [TEMA]:

- 5 de nivel básico (definiciones).
- 10 de nivel intermedio (comprensión y aplicación).
- 5 de nivel avanzado (análisis y síntesis).

Formato de cada tarjeta: pregunta, respuesta breve y explicación.
```

**Para llevarlas a Anki.** Pide el mismo resultado en CSV y lo importas directamente:

```text
Devuélvelo como CSV con tres columnas y separador punto y coma: pregunta;
respuesta; explicación. Sin encabezado y sin numerar las filas.
```

---

## 2. Quiz progresivo

```text
Diseña un quiz de 15 preguntas sobre [TEMA] con dificultad creciente.

Después de cada respuesta mía, proporciona retroalimentación y pasa a la
siguiente pregunta. Si fallo, dame una pista antes de la respuesta correcta.

Hazme una pregunta a la vez y espera mi respuesta.
```

---

## 3. Casos de aplicación

```text
Presenta 5 escenarios prácticos en [ÁREA] para aplicar [TEMA], aumentando la
complejidad gradualmente.

Déjame resolver cada uno antes de dar la solución.
```

---

## 4. Pedir preguntas que valgan la pena

La calidad de las preguntas depende de cómo las pidas, y la diferencia es grande.

**Petición vaga:**

```text
Hazme preguntas sobre la Revolución Francesa.
```

Devuelve casi siempre preguntas de fecha y nombre: el nivel más bajo de la taxonomía de Bloom.

**Petición bien formulada:**

```text
Hazme cinco preguntas sobre la Revolución Francesa que solo pueda responder
si entiendo las causas económicas, y no me des la respuesta hasta que yo
conteste.
```

Devuelve otra cosa: preguntas de análisis, y una conversación en lugar de un cuestionario.

---

## 5. Un prompt por nivel de la taxonomía de Bloom

Cada nivel cognitivo pide un tipo distinto de pregunta. Esta tabla está en el libro; aquí los prompts se pueden copiar.

| Nivel | Qué pide | Prompt |
|---|---|---|
| **Recordar** | Datos, definiciones | `Crea 10 preguntas de opción múltiple sobre fechas y definiciones de [TEMA].` |
| **Comprender** | Explicar con palabras propias | `Formula preguntas sobre [TEMA] que requieran explicar el concepto con mis propias palabras.` |
| **Aplicar** | Resolver con lo aprendido | `Diseña problemas de [TEMA] que requieran aplicar fórmulas o principios a un caso nuevo.` |
| **Analizar** | Comparar, distinguir | `Genera preguntas sobre diferencias y similitudes entre los conceptos de [TEMA].` |
| **Evaluar** | Juzgar con criterio | `Crea casos de [TEMA] donde deba justificar una decisión o evaluar opciones enfrentadas.` |
| **Crear** | Producir algo nuevo | `Plantea desafíos de [TEMA] que requieran diseñar una solución original.` |

**Cómo usar la tabla.** Empieza por «Recordar» solo si el tema es nuevo. Si ya lo estudiaste, entra directo por «Aplicar» o «Analizar»: son los niveles que suele evaluar un examen universitario, y los que más rinden por minuto invertido.
