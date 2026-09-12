# Simular exámenes

Prompts del **capítulo 6**, sección «Simulación de exámenes y preparación evaluativa».

Hacer un simulacro es, además de una forma de prepararse, una forma de práctica de recuperación: la técnica mejor evaluada de las diez que revisan Dunlosky et al. (2013). Y familiarizarse de antemano con el formato quita al examen buena parte de lo que lo hace intimidante.

---

## 1. Examen completo

```text
Crea un examen de [DURACIÓN] minutos sobre [TEMAS] con esta composición:

- 30 % de preguntas de opción múltiple, un punto cada una.
- 40 % de preguntas cortas, tres puntos cada una.
- 30 % un caso práctico o un ensayo, diez puntos.

Total: [X] puntos. Incluye la rúbrica de evaluación.
Nivel: universitario, carrera de [DISCIPLINA].

No me des las respuestas todavía.
```

**Cambia la composición.** Los porcentajes de arriba son un ejemplo. Si conoces el formato real de tu profesor, reprodúcelo: es la mitad del valor del ejercicio.

---

## 2. Análisis posterior

El simulacro sin esta segunda parte sirve de poco.

```text
Estas fueron mis respuestas al examen simulado: [RESPUESTAS]. Por favor:

1. Califícalas según la rúbrica.
2. Identifica los errores conceptuales, no solo los de redacción.
3. Sugiere qué debo repasar para corregir cada error.
4. Proporciona las respuestas modelo.
5. Estima mi calificación probable.
```

---

## 3. Adaptar el simulacro al formato

Cada tipo de evaluación pide una preparación distinta.

### Opción múltiple

```text
Genera 20 preguntas de opción múltiple sobre [TEMA] con cuatro opciones cada
una. Los distractores deben ser plausibles y reflejar errores conceptuales
frecuentes, no respuestas obviamente falsas.

Al final, explica qué error conceptual representa cada distractor.
```

### Ensayo o desarrollo

```text
Dame tres posibles preguntas de desarrollo sobre [TEMA], del tipo que aparece
en un examen universitario.

Para cada una: un esquema de respuesta con los puntos que un evaluador
esperaría encontrar, y la rúbrica con la que los calificaría.
```

### Caso práctico

```text
Crea un caso práctico de [ÁREA] sobre [TEMA], con datos concretos y varias
capas de complejidad.

No me des la solución: hazme preguntas guía que me ayuden a resolverlo por mi
cuenta.
```

### Examen oral

```text
Simula un examen oral de [ASIGNATURA] sobre [TEMAS]. Hazme una pregunta,
espera mi respuesta escrita, y luego dame retroalimentación sobre claridad,
estructura y precisión antes de la siguiente.

Sé exigente con la estructura de la respuesta, no solo con el contenido.
```

---

## 4. Taller de cuatro semanas

Plan completo de preparación para un examen importante.

### Semana 1 — Identificar formato y contenidos

```text
Analiza estos exámenes previos de la asignatura e identifica patrones de
formato, temas recurrentes y nivel de dificultad.

[PEGAR O ADJUNTAR EXÁMENES ANTERIORES]
```

### Semana 2 — Generar material

Crea un banco de al menos treinta preguntas y organízalo por tema y dificultad. Los prompts de [03-recuperacion-activa.md](./03-recuperacion-activa.md) sirven para esto.

### Semana 3 — Simulaciones cronometradas

Tres exámenes simulados, con tiempo real, y análisis de errores después de cada uno. Usa los prompts 1 y 2 de este archivo.

### Semana 4 — Refinamiento

Concéntrate en las áreas débiles con minicuestionarios diarios de diez minutos.

```text
Estos son los temas donde más fallé en los simulacros: [TEMAS].

Hazme un minicuestionario diario de 10 minutos para los próximos 7 días, con
5 preguntas cada uno, concentrado en esos temas. Devuélvelo día por día.
```

### Día previo — Repaso final

```text
Hazme 20 preguntas rápidas de los conceptos más importantes de [TEMAS].
```

---

## 5. Un límite que conviene recordar

Todo esto es preparación **tuya**, con material **tuyo**. Subir a un servicio de IA en la nube los exámenes completos de tus profesores es otra cosa: ver [06-uso-etico-y-citacion.md](./06-uso-etico-y-citacion.md).
