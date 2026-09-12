# Organizar y digitalizar apuntes

Prompts del **capítulo 6**, sección «Optimización y digitalización de apuntes con IA».

De apuntes de clase desordenados a material con el que se puede estudiar: método Cornell, mapas mentales, esquemas y conversión a formatos reutilizables.

---

## 1. Estructuración básica

```text
Te voy a pasar mis apuntes de clase sobre [TEMA]. Por favor:

1. Organízalos en secciones claras.
2. Añade viñetas y numeración donde corresponda.
3. Resalta los conceptos clave en negrita.
4. Crea una tabla de contenidos al inicio.
5. Mantén toda la información original.

[PEGAR APUNTES AQUÍ]
```

---

## 2. Método Cornell (versión completa)

El método Cornell divide la página en tres zonas: notas a la derecha, preguntas de repaso a la izquierda y un resumen abajo. Es la estructura que convierte unos apuntes en algo con lo que puedes autoevaluarte.

```text
Actúa como un asistente de estudio. Convierte las notas que pego al final al
método Cornell y devuélvelo como una tabla de dos columnas, más una fila
final de resumen.

- Columna derecha (notas). Reorganiza las ideas principales en frases
  completas, agrupadas por subtema. No añadas información que no esté en mis
  notas.
- Columna izquierda (preguntas). Por cada bloque de notas escribe una o dos
  preguntas cuya respuesta sea ese bloque. Que pidan comprensión, no
  repetición literal de una definición.
- Resumen final. Cinco líneas como máximo, con lo que debería recordar si
  solo tuviera un minuto.

Si algo de mis notas está incompleto o es ambiguo, márcalo con [VERIFICAR]
en lugar de completarlo tú.

[APUNTES]
```

> **La última línea es la más importante de todo este archivo.** Sin ella, el modelo rellena los huecos de tus apuntes con lo que le parece plausible, y acabas estudiando frases que nunca dijo tu profesor. Con ella, los huecos quedan señalados y sabes exactamente qué tienes que ir a comprobar.

### Ejemplo de resultado

**Lo que había en el cuaderno:**

> mitocondria = ATP. resp celular 3 etapas glucólisis → krebs → cadena transporte e⁻. glucólisis en citoplasma no necesita O₂. krebs en matriz. cadena en membrana interna, aquí la mayoría del ATP ~34. sin O₂ fermentación, poco ATP (2).

**Lo que devolvió la IA:**

| Preguntas | Notas |
|---|---|
| ¿Dónde ocurre cada etapa de la respiración celular? | La respiración celular tiene tres etapas: la glucólisis, en el citoplasma; el ciclo de Krebs, en la matriz mitocondrial; y la cadena de transporte de electrones, en la membrana interna de la mitocondria. |
| ¿Qué etapa aporta la mayor parte del ATP? | La cadena de transporte de electrones, con unas 34 moléculas de ATP **[VERIFICAR: la cifra exacta varía según el texto]**. |
| ¿Qué hace la célula sin oxígeno? | Recurre a la fermentación, que rinde solo 2 ATP. |

**Resumen.** La respiración celular convierte glucosa en ATP en tres etapas y en tres lugares distintos de la célula. El oxígeno es lo que permite llegar a la tercera, que es la que produce casi todo el ATP. Sin oxígeno la célula sobrevive fermentando, con un rendimiento diecisiete veces menor.

Fíjate en dos cosas. La columna de preguntas no se limita a devolver las notas en interrogativo: pregunta *dónde* y *por qué*, que es lo que suele pedir un examen. Y el `[VERIFICAR]` marca un dato que no es cerrado, en lugar de afirmarlo como si lo fuera.

---

## 3. Material complementario a partir de tus apuntes

```text
Basándote en estos apuntes, genera:

1. Un glosario de términos técnicos.
2. Una lista de conceptos para repasar.
3. Cinco preguntas de autoevaluación.
4. Recursos adicionales recomendados.

[APUNTES]
```

---

## 4. Mapa mental

```text
A partir de los apuntes que adjunto al final, crea una estructura de mapa
mental jerárquico con un tema central, subtemas y palabras clave.

Presenta el resultado en formato PlantUML. Dame también las instrucciones
para generar un PNG con una herramienta en línea.

[APUNTES]
```

---

## 5. Diagrama de flujo

```text
Crea un diagrama de flujo en formato PlantUML que muestre el proceso descrito
en estas notas. Usa colores suaves y texto legible.

Dame instrucciones detalladas de cómo generar una imagen PNG con alguna
herramienta en línea.

[APUNTES]
```

Para ver el resultado busca en tu navegador `PlantUML online`, `Mermaid online` o `Graphviz online`, pega el código y descarga la imagen.

---

## 6. Digitalizar apuntes manuscritos

Proceso completo, de la foto al documento.

1. Fotografía tus apuntes con buena iluminación.
2. Extrae el texto con una aplicación de OCR (Google Lens, Office Lens) o directamente con una IA multimodal.
3. Pásalo a la IA con este prompt.
4. Revisa y corrige a mano los errores que queden.
5. Añade el contexto que complemente tus apuntes.

```text
Estos son apuntes escaneados con posibles errores de OCR. Corrige los errores
evidentes, mejora el formato y agrupa por temas relacionados.

Devuélvelo en [LaTeX | Markdown | Word].

Si algo queda ilegible o dudoso, márcalo con [VERIFICAR] en lugar de
adivinar.

[TEXTO OCR]
```

**La calidad de la foto manda.** Iluminación uniforme, contraste y resolución suficiente: si el OCR se equivoca mucho, ninguna IA lo arregla bien después.

---

## 7. Convertir a otros formatos

### Markdown

```text
Convierte estas notas a formato Markdown, con secciones, listas y negritas
para los conceptos clave.

[APUNTES]
```

Para ver el resultado con formato, pega el texto en un editor en línea: [markdownlivepreview.com](https://markdownlivepreview.com/), [stackedit.io](https://stackedit.io/) o [dillinger.io](https://dillinger.io/).

### LaTeX

```text
Convierte este texto con ecuaciones a formato LaTeX, asegurando que las
fórmulas queden bien representadas.

[APUNTES]
```

```text
Genera un documento en LaTeX con mi tarea de cálculo, que adjunto a
continuación como fotos.
```

**Antes de montar un documento largo en Overleaf:** el plan gratuito admite un solo colaborador por proyecto y usa el tiempo de compilación básico, que los planes de pago multiplican por veinticuatro. Una tesis con muchas figuras puede agotarlo. Si te pasa, tienes tres salidas: compilar en tu propio computador, partir el documento en archivos con `\include` para recompilar solo el capítulo en el que trabajas, o pasar a un plan de pago.
