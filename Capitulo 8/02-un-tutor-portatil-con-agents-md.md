# Un tutor portátil: la convención AGENTS.md

Esta guía es para quien preferiría escribir la configuración de su tutor
**una sola vez**, guardarla en un archivo de texto normal dentro de una
carpeta o repositorio, y que esa misma configuración funcione en varias
herramientas de IA sin copiar y pegar de nuevo en cada una. No sustituye
la ruta de Proyectos de [`01-configurar-tu-tutor-en-chatgpt-y-claude.md`](./01-configurar-tu-tutor-en-chatgpt-y-claude.md)
--- es una alternativa para quien ya usa (o quiere empezar a usar) alguna
de las herramientas de agente que trabajan directamente sobre archivos.

## Qué es AGENTS.md

`AGENTS.md` es una convención abierta: un archivo de texto plano, en la
raíz de un proyecto o carpeta, con instrucciones para un agente de IA que
tiene acceso a esos archivos. Más de 30 herramientas de agentes lo leen de
forma nativa, entre ellas Codex de OpenAI (la base de los modos de agente
de ChatGPT para tareas de código), Cursor, Aider, GitHub Copilot y varias
más. La idea es simple: en lugar de repetirle a cada herramienta quién
eres y cómo quieres que trabaje, lo escribes una vez en `AGENTS.md` y
cualquier herramienta compatible lo lee automáticamente al abrir esa
carpeta.

Nació para configurar agentes que escriben y revisan código, así que la
mayoría de sus ejemplos hablan de comandos de compilación y convenciones
de estilo. Nada impide reutilizar el mismo mecanismo para un caso de uso
distinto: un tutor de estudio que vive dentro de una carpeta de apuntes y
ejercicios, versionada con git como cualquier otro proyecto.

## El detalle que hay que conocer antes de usarlo: Claude Code lee CLAUDE.md

Aquí hay un matiz importante, verificado directamente en la documentación
de Anthropic: Claude Code --- y, por lo tanto, las sesiones de **Claude
Cowork** que se ejecutan sobre él en la aplicación de escritorio --- no
lee `AGENTS.md` directamente. Lee un archivo llamado `CLAUDE.md`.

La propia documentación de Anthropic resuelve esto con un patrón oficial
de dos líneas: se crea un `CLAUDE.md` que simplemente importa el contenido
de `AGENTS.md` con la sintaxis `@AGENTS.md`. Con eso, ambas herramientas
--- las que leen `AGENTS.md` de forma nativa y Claude Code/Cowork, que lee
`CLAUDE.md` --- terminan usando exactamente las mismas instrucciones, sin
mantener dos copias.

Los archivos [`plantillas/AGENTS.md`](./plantillas/AGENTS.md) y
[`plantillas/CLAUDE.md`](./plantillas/CLAUDE.md) de esta carpeta ya
implementan este patrón: el segundo es literalmente una línea que importa
al primero.

## Cómo montarlo, paso a paso

1. **Crea una carpeta** (o un repositorio de git, si quieres historial de
   cambios) para tu materia: `tutor-calculo/`, por ejemplo.
2. **Organízala** con subcarpetas para tus apuntes, tus ejercicios y tus
   documentos de referencia (sílabo, rúbricas). El ejemplo de
   [`plantillas/AGENTS.md`](./plantillas/AGENTS.md) asume `apuntes/`,
   `ejercicios/` y `recursos/`, pero los nombres son arbitrarios.
3. **Copia `AGENTS.md` y `CLAUDE.md`** de la carpeta `plantillas/` a la
   raíz de tu carpeta, y adapta el contenido de `AGENTS.md` a tu materia
   y nivel (los corchetes indican qué reemplazar). No necesitas tocar
   `CLAUDE.md`: ya está listo.
4. **Abre esa carpeta con la herramienta de agente que prefieras.** Si
   usas una herramienta que trabaja con `AGENTS.md` de forma nativa
   (Codex, por ejemplo), la leerá directamente. Si usas Claude Code o una
   sesión de Claude Cowork sobre esa carpeta, leerá `CLAUDE.md`, que a su
   vez importa el mismo contenido.
5. **Interactúa normalmente**, pidiendo ayuda con un ejercicio de la
   carpeta `ejercicios/`. El agente debería comportarse como el tutor
   descrito en `AGENTS.md`, no como un asistente de programación
   genérico, incluso si la herramienta que usas fue diseñada
   originalmente para escribir código.

## Por qué esto le sirve a un lector no técnico

Si esto suena a más esfuerzo del necesario para simplemente hablar con un
tutor de IA, tienes razón: para el uso más común (una conversación de
estudio puntual), la ruta de Proyectos de la guía anterior es más simple
y no requiere ninguna herramienta de línea de comandos. Esta ruta tiene
sentido cuando:

- Ya llevas tus apuntes o ejercicios en una carpeta versionada con git
  (algo común entre estudiantes de programación, ingeniería o ciencias
  de datos) y quieres que el tutor viva ahí mismo, junto al material.
- Quieres una sola fuente de verdad para la configuración de tu tutor,
  reutilizable si cambias de herramienta más adelante, en lugar de
  reescribirla en la configuración de cada plataforma por separado.
- Colaboras con colegas (otros docentes de la misma materia, por
  ejemplo) y quieres compartir y versionar la configuración del tutor
  igual que cualquier otro archivo del curso, con historial de cambios
  incluido.

## Vigencia

El soporte de herramientas concretas para `AGENTS.md`, y el mecanismo de
puente de Claude Code hacia `CLAUDE.md`, se verificaron por última vez en
**septiembre de 2026** contra la documentación oficial de cada
herramienta. La convención en sí --- un archivo de texto con
instrucciones que un agente lee al abrir una carpeta --- es
suficientemente simple como para que sobreviva a cambios de nombre o de
versión en herramientas específicas.
