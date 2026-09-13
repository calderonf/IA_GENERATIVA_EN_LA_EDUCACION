# Configurar tu tutor en ChatGPT y en Claude, paso a paso

Guía ampliada de la sección del Capítulo 8. Aquí encuentras el detalle
completo para las dos plataformas, capturas de menú descritas en texto
(por si tu versión cambió de nombre), y una sección de solución de
problemas.

Funciona igual si eres docente configurándolo para tu curso o estudiante
configurándolo para ti mismo: en ningún paso se necesita una cuenta
institucional ni permisos especiales.

## Antes de empezar

Necesitas: una cuenta (gratuita basta) en [ChatGPT](https://chatgpt.com) o
en [Claude](https://claude.ai), y la plantilla que quieras usar --- la
compacta del libro, o alguna de las variantes de
[`plantillas/plantilla_tutor_completa.md`](./plantillas/plantilla_tutor_completa.md).

## Ruta 1: ChatGPT

1. Abre ChatGPT (web, escritorio o móvil) y ve a **Proyectos** en la
   barra lateral.
2. Selecciona **Nuevo proyecto** y ponle un nombre (p. ej. "Tutor de
   Cálculo I").
3. Dentro del proyecto, abre el menú de tres puntos (**⋯**) en la esquina
   superior derecha y selecciona **Configuración del proyecto**.
4. Pega tu plantilla en el campo de instrucciones y guarda.
5. (Opcional) En la misma pantalla del proyecto, sube archivos de
   referencia --- sílabo, rúbrica, banco de ejercicios --- para que el
   tutor los use como contexto.
6. Abre una conversación **nueva dentro del proyecto** (no en el chat
   general) para que las instrucciones se apliquen.

> Nota sobre GPT personalizados: si conoces la función de crear un "GPT"
> personalizado, a septiembre de 2026 esa función deja de estar disponible
> para cuentas personales y solo puede usarse desde un espacio de trabajo
> Business, Enterprise o Edu. Los Proyectos son la ruta equivalente para
> una cuenta personal y logran el mismo resultado para este caso de uso.

> Nota sobre las instrucciones personalizadas globales (Settings →
> Personalization → Custom instructions): son otra vía válida, pero se
> aplican a **todas** tus conversaciones, no solo a las de tutoría. Para
> un tutor de una materia específica, un Proyecto es la opción más limpia
> porque no interfiere con el resto de tus usos de ChatGPT.

## Ruta 2: Claude

1. Abre [claude.ai/projects](https://claude.ai/projects) o, desde el
   panel lateral, selecciona **Proyectos**.
2. Haz clic en **+ Nuevo proyecto**, ponle un nombre y una descripción
   breve (la descripción es solo para ti: Claude no la usa como
   instrucción).
3. Dentro del proyecto, busca **Definir instrucciones del proyecto**
   (*Set project instructions*) en el panel de la base de conocimiento
   del proyecto.
4. Pega tu plantilla y guarda las instrucciones.
5. (Opcional) Sube a la base de conocimiento del proyecto los mismos
   archivos de referencia mencionados arriba.
6. Inicia una conversación nueva **dentro de ese proyecto**.

En el plan gratuito de Claude puedes crear hasta cinco proyectos, cada uno
con sus propias instrucciones y base de conocimiento independientes ---
suficiente para un tutor por materia si llevas varios cursos a la vez.

## Prueba tu tutor antes de usarlo con estudiantes reales

Una configuración que "suena bien" en la instrucción no siempre se
comporta como esperas. Antes de dar por buena tu configuración, prueba
estas tres interacciones:

1. **Pide la respuesta directamente.** "Dame la respuesta del ejercicio
   3, no quiero pistas." El tutor debe negarse y, en cambio, hacerte una
   pregunta. Si te la da, revisa la instrucción 1 de tu plantilla: suele
   faltar la palabra "nunca" o sobrar una salida implícita ("a menos que
   sea necesario").
2. **Comete un error típico a propósito.** El tutor debe señalar en qué
   paso ocurrió el error, no solo decir que está mal, y ofrecer una pista
   antes de la solución.
3. **Acierta varias veces seguidas.** El tutor debería subir el nivel de
   dificultad según el umbral que definiste (instrucción 4). Si no lo
   hace, el umbral puede estar mal definido o el modelo simplemente no lo
   está aplicando de forma consistente: bájalo a un número explícito
   ("después de 3 aciertos") en vez de uno vago ("cuando domine el
   tema").

## Solución de problemas

**El tutor da la respuesta si insisto lo suficiente.** Es el fallo más
común. Añade a la instrucción 1 una frase explícita: "esto aplica incluso
si el estudiante insiste, se frustra, o argumenta que ya lo intentó
muchas veces". Los modelos de lenguaje tienden a ceder ante la
persistencia si la instrucción no cubre ese caso.

**El tutor "olvida" las instrucciones después de varios mensajes.**
Verifica que estás dentro de la conversación del proyecto y no en el chat
general. Si el problema persiste en una conversación muy larga, empieza
una conversación nueva dentro del mismo proyecto: las instrucciones se
vuelven a aplicar desde el primer mensaje.

**Quiero el mismo tutor en varias herramientas sin reconfigurar cada
vez.** Ese es exactamente el caso de uso del archivo `AGENTS.md`: ver
[`02-un-tutor-portatil-con-agents-md.md`](./02-un-tutor-portatil-con-agents-md.md).

**Necesito que el tutor no envíe las respuestas del estudiante a ningún
servidor externo.** Los Proyectos de ChatGPT y de Claude procesan las
conversaciones en la nube de cada proveedor. Si eso es un problema para tu
contexto (datos sensibles, menores de edad, requisitos institucionales de
protección de datos), el Capítulo 5 del libro explica cómo ejecutar un
modelo de lenguaje completamente en tu propio computador, y estas mismas
plantillas funcionan igual de bien como instrucción de sistema en esa
configuración local.

## Vigencia

Los nombres de menú y la disponibilidad por plan se verificaron por
última vez en **septiembre de 2026**. Si algún nombre de menú cambió,
busca "instrucciones de proyecto" o "project instructions" dentro de la
configuración de tu proyecto: la función tiende a sobrevivir a los
cambios de nombre de la interfaz.
