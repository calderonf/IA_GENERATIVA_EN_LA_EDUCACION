# Capítulo 8 · Personalización del aprendizaje y tutorías IA

Material complementario del libro **«Inteligencia artificial generativa
en la Educación. Una guía para Estudiantes, Docentes e Instituciones»**.

Todo lo que necesitas para pasar del diseño pedagógico del capítulo a un
tutor de IA que funciona de verdad, hoy, sin escribir código: guías
paso a paso, una plantilla completa con variantes por disciplina, y una
alternativa portátil para quien prefiera un archivo de configuración
reutilizable entre varias herramientas.

Sirve igual si eres docente configurando un tutor para tu curso o
estudiante configurando uno para ti mismo.

---

## Qué hay aquí

| Archivo | Para qué |
|---|---|
| [01 · Configurar tu tutor en ChatGPT y en Claude](./01-configurar-tu-tutor-en-chatgpt-y-claude.md) | Guía completa paso a paso de la sección del capítulo, con las dos plataformas en detalle, cómo probar tu configuración antes de usarla con estudiantes reales, y solución de problemas comunes. |
| [02 · Un tutor portátil con AGENTS.md](./02-un-tutor-portatil-con-agents-md.md) | Para quien prefiera una configuración reutilizable entre herramientas en lugar de reescribirla en cada plataforma: qué es la convención `AGENTS.md`, por qué Claude Code/Cowork necesita un archivo `CLAUDE.md` adicional, y cómo montarlo. |

### Plantillas

| Plantilla | Para qué |
|---|---|
| [Plantilla completa del tutor](./plantillas/plantilla_tutor_completa.md) | La versión extendida de la plantilla compacta del libro, con variantes ya redactadas para matemáticas, ciencias naturales, historia/ciencias sociales, lengua y escritura, y programación. |
| [AGENTS.md](./plantillas/AGENTS.md) | Archivo de ejemplo, listo para adaptar, que configura el tutor con la convención AGENTS.md descrita en la guía 02. |
| [CLAUDE.md](./plantillas/CLAUDE.md) | Archivo puente de una línea para que Claude Code y Claude Cowork lean las mismas instrucciones que `AGENTS.md`, sin duplicarlas. |

---

## Cómo usar los prompts y plantillas

Las plantillas están en bloques de código para que las copies enteras,
sin arrastrar el formato del texto que las rodea. Lo que va **entre
corchetes** --- `[MATERIA]`, `[NIVEL]`, `[N]` --- son marcadores que debes
reemplazar antes de guardar la configuración.

Ninguna plantilla necesita conocimientos de programación para usarse en
la ruta de Proyectos (guía 01). La ruta de `AGENTS.md` (guía 02) sí asume
cierta familiaridad con carpetas versionadas con git y con al menos una
herramienta de agente de IA basada en archivos.

---

## Por qué el capítulo insiste en que nunca dé la respuesta directa

Las seis reglas que se repiten en todas las plantillas de esta carpeta no
son una preferencia de estilo: son la traducción operativa de los
principios pedagógicos del capítulo --- diagnóstico inicial, diálogo
socrático, pistas escalonadas, avance por dominio (*mastery learning*) y
cierre metacognitivo. Un tutor de IA que responde directamente a la
primera pregunta deja de ser un tutor y vuelve a ser, simplemente, un
buscador de respuestas. La regla 1 de cada plantilla existe precisamente
para que eso no pase, incluso cuando el estudiante presiona para que
pase.

Antes de desplegar cualquiera de estas plantillas con estudiantes
menores de edad de forma institucional, revisa la sección sobre
evaluación de impacto y protección de datos del propio capítulo: aplica
directamente a un tutor configurado con este material.

---

## Vigencia

Los nombres de plataformas, menús y planes gratuitos que aparecen en este
material se verificaron por última vez en **septiembre de 2026**. Como en
el resto del repositorio, los principios pedagógicos de las plantillas
envejecen mucho mejor que los nombres de las herramientas: si un menú
cambió de nombre, busca "instrucciones de proyecto" o "project
instructions" y la función casi siempre sigue ahí.
