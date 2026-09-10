# Capítulo 4 — Materiales complementarios

Complementos del **capítulo 4** de *Inteligencia artificial generativa en la Educación*: herramientas de IA generativa gratuitas en línea y cómo citarlas.

| Archivo | Qué es |
|---|---|
| [`Lista_verificacion_herramientas.md`](./Lista_verificacion_herramientas.md) | Lista de verificación completa y plantilla de registro para comprobar una herramienta antes de recomendarla a un estudiante. Versión extendida de la que aparece resumida en el libro. |
| [`TablaComparativaIAGGratuita/`](./TablaComparativaIAGGratuita) | Tabla comparativa en LaTeX, lista para adaptar, con el PDF ya compilado. Verificada el 10-sep-2026. |
| `Comparativa_Modelos_IA.pptx` | Presentación de apoyo, 11 diapositivas. Verificada el 10-sep-2026. |
| [`Retos con IA Generativa en Parejas/`](./Retos%20con%20IA%20Generativa%20en%20Parejas) | Actividad de aula en parejas, con su banco de imágenes. |

## Comparativa de planes gratuitos

**Verificada el 10 de septiembre de 2026 contra la documentación oficial de cada proveedor.**

Esta tabla responde a las preguntas que un docente necesita resolver antes de mandar a un grupo a usar una herramienta. No pretende decir cuál es «la mejor»: eso depende de tu asignatura y caduca en meses.

| Herramienta | Plan gratuito | Búsqueda web en el gratuito | Genera imágenes en el gratuito | Ejecuta código en el gratuito | Datos del gratuito usados para entrenar | Desde (USD/mes) |
|---|---|---|---|---|---|---|
| **ChatGPT** (OpenAI) | Sí, con límites | **Sí** ✅ | Sí, con límites | Sí, con límites | Sí por defecto; exclusión disponible | Go 8 · Plus · Pro |
| **Claude** (Anthropic) | Sí, con límites | **Sí** ✅ | No | **Sí** ✅ | No por defecto; requiere consentimiento | Pro 20 (17 anual) |
| **Gemini** (Google) | Sí, con límites | Sí | **Sí**, incluida edición | Sí | Sí por defecto; desactivarlo cuesta el historial | AI Plus 4,99 · AI Pro 19,99 |
| **Mistral** (Vibe) | Sí, con límites | Sí, limitada | **Sí** | Sí, sesiones limitadas | Sí, con exclusión en la cuenta | Pro 14,99 · estudiante 5,99 |
| **Grok** (xAI) | Sí | Sí | Limitada | — | No por defecto en grok.com; sí dentro de X | SuperGrok 30 (300 anual) |
| DeepSeek | Sí | Sí | Sí (modelos Janus) ᴰ | *sin verificar* | Sí por defecto; exclusión poco visible | — |
| Qwen (Qwen Studio) | Sí | *no documentada* | Sí (Qwen-Image) ᴰ | *sin verificar* | *documentación limitada* | — |
| Kimi (Moonshot) | Sí | *sin verificar* | No documentada | *sin verificar* | *sin mecanismo claro de exclusión* | — |
| Meta AI | Sí | Sí | Sí | No | Sí; exclusión compleja y sin garantía | — |

### Tres cosas que conviene saber antes de recomendar

**El plan gratuito ya no significa lo mismo en todas partes.** OpenAI anunció que empezará a probar publicidad en sus planes gratuito y Go para usuarios con sesión iniciada; los planes Pro, Business y Enterprise no la incluyen. Y en casi todos los proveedores, el plan gratuito es justamente aquel en el que los datos se usan para entrenar por defecto.

**Aparecieron planes intermedios de bajo costo.** ChatGPT Go y Google AI Plus se dirigen al estudiante que agota el plan gratuito pero no puede pagar una suscripción completa. Suelen ser la recomendación más sensata para ese caso.

**Mistral renombró su asistente a Vibe.** «Le Chat» sobrevive solo como nombre de la aplicación móvil en las tiendas.

### Cuatro herramientas: capacidades sí, condiciones comerciales no

DeepSeek, Qwen, Kimi y Meta AI aparecen con varias celdas marcadas como *sin verificar*, y conviene precisar qué falta exactamente, porque no es todo.

**Sus capacidades sí están documentadas oficialmente**, y las fuentes están más abajo en el nivel 1: DeepSeek publica los modelos Janus en su repositorio y su ficha de modelo; Qwen documenta Qwen-Image en su blog de ingeniería; Moonshot publica la documentación del modelo de visión de Kimi; Meta describe en sus centros de ayuda cómo generar imágenes y preguntar por una foto en el chat. Las celdas marcadas con **ᴰ** se apoyan en esa documentación oficial, aunque sea de 2025.

**Lo que no pudimos establecer son las condiciones comerciales**: qué se ofrece exactamente en el plan gratuito, con qué límites, a qué precio y bajo qué política de datos. Que un modelo exista y esté documentado no significa que esté disponible sin pagar, ni que lo siga estando el próximo semestre.

| Herramienta | Obstáculo encontrado el 10-sep-2026 |
|---|---|
| Meta AI | El sitio **impide la consulta automatizada** mediante `robots.txt` |
| DeepSeek | La portada no publica precios ni capacidades del servicio de consumo |
| Qwen | Ahora **Qwen Studio**. Enumera capacidades pero no precios, y no menciona búsqueda web |
| Kimi | Página en chino, sin precios accesibles |

Preferimos dejar el hueco visible antes que rellenarlo con datos de terceros sin comprobar.

La distinción importa al decidir. Puedes explicarle a un estudiante qué sabe hacer una de estas herramientas apoyándote en documentación oficial; lo que no puedes es garantizarle que la tendrá gratis, ni asegurarle a tu institución qué se hará con lo que suba. **Cuando no es posible determinar qué plan se está usando ni qué ocurre con los datos, esa opacidad es en sí misma parte del riesgo de adoptar la herramienta.**

---

## Comparativa detallada de 2025 (archivo)

La tabla que sigue se elaboró en **2025** y se conserva porque entra en más detalle que la de arriba: distingue entrada frente a generación de imágenes, modos de investigación profunda y modos de estudio. **No se ha reverificado.** Contrástala siempre con la página oficial antes de recomendar nada.

| Servicio (gratuito)                                  | ¿Recibe **imágenes**?                                                                                                                                | ¿**Genera** imágenes?                                                                                                                                     | ¿“Deep search / deep research”?                                                                                                                                           | ¿Modo **estudio** (tipo “estudia y aprende”)?                 | **Búsqueda web**                                                                                      | **Búsqueda en documentos**                                                                                   |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **ChatGPT (OpenAI)**                                 | **Limitado**: la ayuda de OpenAI indica que “image inputs” están en Plus/Enterprise; los *archivos* sí están en Free (límite diario) ([BytePlus][1]) | **Sí** (4o image generation disponible también para Free)                                                                                                 | **Sí** (Deep Research) con cupo limitado en Free según lanzamiento/actualizaciones ([X (formerly Twitter)][2])                                                            | **Sí** (Study Mode) disponible en la app, con límites en Free | **Sí** (ChatGPT Search; disponible incluso sin iniciar sesión) ([BytePlus][3])                        | **Sí** (subir y analizar archivos en Free, con límites) ([Analytics Vidhya][4])                              |
| **Gemini (Google)**                                  | **Sí** (entrada multimodal) ([Google AI for Developers][5])                                                                                          | **Sí** (“Nano Banana”/Gemini Image; generar y **editar** imágenes en Apps) ([Google Ayuda][6])                                                            | **Sí** (Gemini **Deep Research**, “pruébalo sin costo”) ([Gemini][7])                                                                                                     | —                                                             | **Sí** (integrado) ([Gemini][8])                                                                      | **Sí** (desde feb-2025 los **free** pueden subir y analizar Docs/PDF, y también vía Drive) ([9to5Google][9]) |
| **Claude (Anthropic)**                               | **Sí** (puedes subir imágenes) ([Centro de Ayuda Anthropic][10])                                                                                     | **No** (no produce imágenes) ([Centro de Ayuda de Anthropic][11])                                                                                         | **No** como “modo” aparte; **sí** tiene **Web Search** global en todas las cuentas (incl. Free) ([Anthropic][12])                                                         | —                                                             | **Sí** (Web Search) ([Anthropic][12])                                                                 | **Sí** (subida de archivos en Free; límites por tamaño/cantidad) ([Centro de Ayuda Anthropic][13])           |
| **DeepSeek**                                         | **Sí** (modelos Janus para visión) *en app puede variar* ([GitHub][14])                                                                              | **Sí** (familia Janus para generación de imágenes; a veces vía terceros) ([Hugging Face][15])                                                             | **Sí** (chat con búsqueda web en su sitio, con límites) ([DeepSeek][16])                                                                                                  | —                                                             | **Sí** (tiempo real/“search”) ([DeepSeek][16])                                                        | **Sí** (lectura de archivos en el chat) ([DeepSeek][17])                                                     |
| **Grok (xAI)**                                       | **Sí** (capacidad de comprensión de imágenes en Grok 3/4) ([xAI][18])                                                                                | **Limitado**: **Grok Imagine** (imagen/video) existe pero el acceso pleno suele ser **de pago** (SuperGrok/Premium+); free con recortes ([The Verge][19]) | **Sí** (DeepSearch/DeeperSearch; con cuotas en free) ([Reddit][20])                                                                                                       | —                                                             | **Sí** (búsqueda en tiempo real integrada) ([xAI][21])                                                | —                                                                                                            |
| **Qwen Chat (Alibaba)**                              | **Sí** (imagen y **video** understanding) ([Qwen Chat][22])                                                                                          | **Sí** (Qwen-Image; gratuito en chat) ([Qwen][23])                                                                                                        | —                                                                                                                                                                         | —                                                             | **Sí** (integrado) ([Qwen Chat][22])                                                                  | **Sí** (procesamiento de documentos) ([Qwen][24])                                                            |
| **Kimi (Moonshot AI)**                               | **Sí** (visión; Kimi Vision/API y app multimodal) ([platform.moonshot.ai][25])                                                                       | **No confirmado** en la app de consumo (Kimi destaca **visión/entendimiento**, no un generador propio) ([platform.moonshot.ai][25])                       | **Sí**: búsqueda en línea; y **Kimi Researcher** (agente de investigación) aparece como función del servicio (a menudo con plan de pago/uso limitado) ([moonshot.ai][26]) | —                                                             | **Sí** (online search) ([moonshot.ai][26])                                                            | **Sí** (muy largo contexto y manejo de archivos; detalles en reseñas) ([Cursor IDE中文站][27])                  |
| **Meta AI (meta.ai / Instagram/WhatsApp/Messenger)** | **Sí** (puedes **preguntar por una foto** que envías al chat) ([Centro de ayuda de Instagram][28])                                                   | **Sí** (generar y **editar** imágenes/GIFs gratis en sus apps y web) ([Meta][29])                                                                         | —                                                                                                                                                                         | —                                                             | **Sí** (“capacidad de **buscar a través de la web**” en la app de Meta AI) ([Acerca de Facebook][30]) | —                                                                                                            |

**Notas rápidas y matices importantes**

* En **ChatGPT Free**, *cargar archivos* está permitido con cupos; **“image input”** como tal aparece en la ayuda para **Plus/Enterprise**. Para tus actividades con imágenes en ChatGPT Free, mejor usa **generación** (sí disponible) y **Search** para la parte de web. ([Analytics Vidhya][4])
* **Gemini** hoy es el más “todo-en-uno” gratis: visión, **generación/edición** de imágenes (“Nano Banana”), **Deep Research** y **análisis de documentos** en cuentas gratuitas (recientemente habilitado). Ideal para los equipos que harán retos multimodales. ([Gemini][7])
* **Claude** no genera imágenes, pero es excelente en **análisis** de imágenes/documentos + **Web Search** con citas (gratis). Perfecto para el grupo “investigación con fuentes”. ([Centro de Ayuda de Anthropic][11])
* **Grok** tiene **DeepSearch** muy motivante para los chicos, pero su **generación** visual potente (**Grok Imagine**) suele requerir suscripción; el plan gratuito tiene límites de uso. ([The Verge][19])
* **Kimi** trae **Researcher** (tipo “deep research”) y **visión**; la generación de imágenes en el cliente público no está documentada como disponible. Disponibilidad puede variar por región/idioma. ([Wikipedia][31])
* **Meta AI** es muy útil si tus estudiantes ya usan WhatsApp/Instagram: **subes una foto y preguntas**, o pides que **genere** imágenes en el mismo chat. También “mira” la web. ([Centro de ayuda de Instagram][28])

---

## Fuentes

Las referencias siguientes sustentan la comparativa de 2025. Las separamos por nivel, que es exactamente el criterio que enseña el capítulo 4 del libro: **una afirmación vale lo que vale su fuente**.

### Nivel 1 — Documentación oficial del proveedor

Son fuentes primarias: documentación técnica, centros de ayuda, salas de prensa y repositorios de las propias empresas. Es lo que hay que citar en un trabajo académico y lo que hay que consultar antes de recomendar una herramienta.

**Google / Gemini**
- [Image generation with Gemini (Nano Banana) — Gemini API](https://ai.google.dev/gemini-api/docs/image-generation)
- [Generate & edit images with Gemini Apps](https://support.google.com/gemini/answer/14286560?co=GENIE.Platform%3DDesktop&hl=en)
- [Gemini Deep Research](https://gemini.google/overview/deep-research/)
- [Google Gemini](https://gemini.google.com/)

**Anthropic / Claude**
- [Claude — Centro de ayuda](https://support.anthropic.com/en/collections/4078531-claude)
- [Can Claude produce images?](https://support.claude.com/en/articles/9002504-can-claude-produce-images)
- [Claude can now search the web](https://www.anthropic.com/news/web-search)
- [Getting started with Claude](https://support.anthropic.com/en/articles/8114491-getting-started-with-claude)

**DeepSeek**
- [Janus-Series: Unified Multimodal Understanding — repositorio oficial](https://github.com/deepseek-ai/Janus)
- [deepseek-ai/Janus-Pro-7B — model card](https://huggingface.co/deepseek-ai/Janus-Pro-7B)
- [DeepSeek](https://www.deepseek.com/en)
- [DeepSeek Chat](https://chat.deepseek.com/)

**xAI / Grok**
- [Grok 3 Beta — The Age of Reasoning Agents](https://x.ai/news/grok-3)
- [xAI](https://x.ai/)

**Alibaba / Qwen**
- [Qwen Chat](https://chat.qwen.ai/)
- [Qwen-Image: Crafting with Native Text Rendering — blog oficial](https://qwenlm.github.io/blog/qwen-image/)
- [Download Qwen](https://qwen.ai/download)
- [Anuncio de Deep Research en Qwen Chat — cuenta oficial de Qwen](https://x.com/Alibaba_Qwen/status/1958506067324960816)

**Moonshot AI / Kimi**
- [Use the Kimi Vision Model — documentación oficial](https://platform.moonshot.ai/docs/guide/use-kimi-vision-model)
- [Moonshot AI](https://www.moonshot.ai/)

**Meta**
- [Ask Meta AI about an image you share in a chat on Instagram](https://help.instagram.com/3820477441501878/?helpref=related_articles)
- [Generate images using Meta AI — Meta Help Center](https://www.meta.com/help/artificial-intelligence/1337455336906126/)
- [Introducing the Meta AI App](https://about.fb.com/news/2025/04/introducing-meta-ai-app-new-way-access-ai-assistant/)

### Nivel 2 — Prensa especializada

Medios con redacción y verificación editorial. Sirven para fechar un lanzamiento o entender un contexto, pero para un dato concreto —un precio, un límite— conviene contrastar con la fuente oficial.

- [xAI's new Grok image and video generator has a 'spicy' mode — The Verge](https://www.theverge.com/news/718795/xai-grok-imagine-video-generator-spicy-mode)
- [Free Gemini users can now upload, analyze documents — 9to5Google](https://9to5google.com/2025/02/25/free-gemini-document-upload/)

### Nivel 3 — Fuentes de tercera mano: úsalas con cautela

Blogs comerciales, foros y enciclopedias. **No son citables en un trabajo académico** y no deben sustentar por sí solas una decisión institucional. Las conservamos porque en su momento sirvieron como pista —a veces son lo único que documenta un límite del plan gratuito que la empresa no publica—, pero cada dato que provenga de aquí debe verificarse contra el nivel 1 antes de darlo por bueno.

- [How to Use Qwen 2.5 Max AI Online for Free in 2025 — BytePlus](https://www.byteplus.com/en/topic/418442)
- [Qwen Chat DeepSearch AI — BytePlus](https://www.byteplus.com/en/topic/418429)
- [Qwen-Image: Alibaba's Free Image Generation Model — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2025/08/qwen-image/)
- [Kimi AI Review 2025 — Cursor IDE](https://www.cursor-ide.com/blog/kimi-ai-review-2025)
- [Free Tier Limits (as of 2nd April 2025) — r/grok](https://www.reddit.com/r/grok/comments/1jpjmy6/free_tier_limits_as_of_2nd_april_2025/)
- [Kimi (chatbot) — Wikipedia](https://en.wikipedia.org/wiki/Kimi_%28chatbot%29)

### Una nota sobre este listado

Ninguna fuente es perfecta, y la clasificación de arriba no es un juicio sobre su honestidad: es una escala de **cuánto peso puede soportar cada una**. Incluso la documentación oficial envejece —una página de precios de 2025 puede describir un plan que ya no existe— y a veces omite justo lo que necesitas saber, que es la razón por la que a menudo el único registro de un límite concreto está en un foro.

El capítulo 4 del libro desarrolla esto en su **protocolo de verificación en cuatro pasos**: solicitar fuentes, comprobar que existen y dicen lo que se afirma, triangular con al menos dos fuentes independientes, y documentar el proceso. La regla práctica: **cuanto más consecuencias tenga la decisión, más arriba en esta escala debe estar la fuente que la sustenta.** Para probar una herramienta en tu propia clase, una pista de un foro puede bastarte. Para redactar la política de IA de una facultad, no.

Los enlaces se conservan sin el parámetro de rastreo `?utm_source=` con el que fueron recopilados.

---

<!-- Definiciones de las citas numeradas que aparecen en la comparativa de 2025.
     En el archivo original estaban como lista numerada, asi que las citas [Nombre][N]
     del texto no resolvian a ningun enlace. Convertidas al formato de referencia de
     Markdown para que funcionen, y limpiadas del parametro de rastreo utm_source.
     La clasificacion por niveles de fiabilidad esta en la seccion Fuentes, mas arriba. -->

[1]: https://www.byteplus.com/en/topic/418442 "How to Use Qwen 2.5 Max AI Online for Free in 2025"
[2]: https://x.com/Alibaba_Qwen/status/1958506067324960816 "Discover Deep Research in Qwen Chat for free!"
[3]: https://www.byteplus.com/en/topic/418429 "Qwen Chat DeepSearch AI. Advanced AI Assistant 2025"
[4]: https://www.analyticsvidhya.com/blog/2025/08/qwen-image/ "Qwen-Image. Alibaba's Free Image Generation Model is ..."
[5]: https://ai.google.dev/gemini-api/docs/image-generation "Image generation with Gemini (aka Nano Banana) - Gemini API"
[6]: https://support.google.com/gemini/answer/14286560?co=GENIE.Platform%3DDesktop&hl=en "Generate & edit images with Gemini Apps - Computer"
[7]: https://gemini.google/overview/deep-research/ "Gemini Deep Research — your personal research assistant"
[8]: https://gemini.google.com/ "Google Gemini"
[9]: https://9to5google.com/2025/02/25/free-gemini-document-upload/ "Free Gemini users can now upload, analyze documents U"
[10]: https://support.anthropic.com/en/collections/4078531-claude "Claude | Anthropic Help Center"
[11]: https://support.claude.com/en/articles/9002504-can-claude-produce-images "Can Claude produce images? - Anthropic Help Center"
[12]: https://www.anthropic.com/news/web-search "Claude can now search the web"
[13]: https://support.anthropic.com/en/articles/8114491-getting-started-with-claude "Getting started with Claude | Anthropic Help Center"
[14]: https://github.com/deepseek-ai/Janus "Janus-Series: Unified Multimodal Understanding and ..."
[15]: https://huggingface.co/deepseek-ai/Janus-Pro-7B "deepseek-ai/Janus-Pro-7B"
[16]: https://www.deepseek.com/en "DeepSeek"
[17]: https://chat.deepseek.com/ "DeepSeek"
[18]: https://x.ai/news/grok-3 "Grok 3 Beta — The Age of Reasoning Agents"
[19]: https://www.theverge.com/news/718795/xai-grok-imagine-video-generator-spicy-mode "xAI's new Grok image and video generator has a 'spicy' mode"
[20]: https://www.reddit.com/r/grok/comments/1jpjmy6/free_tier_limits_as_of_2nd_april_2025/ "Free Tier Limits (as of 2nd April 2025) - grok"
[21]: https://x.ai/ "xAI: Welcome"
[22]: https://chat.qwen.ai/ "Qwen Chat"
[23]: https://qwenlm.github.io/blog/qwen-image/ "Qwen-Image: Crafting with Native Text Rendering"
[24]: https://qwen.ai/download "Download Qwen"
[25]: https://platform.moonshot.ai/docs/guide/use-kimi-vision-model "Use the Kimi Vision Model - Moonshot AI Open Platform"
[26]: https://www.moonshot.ai/ "Moonshot AI"
[27]: https://www.cursor-ide.com/blog/kimi-ai-review-2025 "Kimi AI Review 2025. 2 Million Character Context ... - Cursor IDE"
[28]: https://help.instagram.com/3820477441501878/?helpref=related_articles "Ask Meta AI about an image you share in a chat on Instagram"
[29]: https://www.meta.com/help/artificial-intelligence/1337455336906126/?srsltid=AfmBOoqo5457O5J2Xu7mupXKCYU-JoqEjaiWWYAOAsGuuhxzYajGnIqG "Generate images using Meta AI | Meta Help Center"
[30]: https://about.fb.com/news/2025/04/introducing-meta-ai-app-new-way-access-ai-assistant/ "Introducing the Meta AI App: A New Way to Access Your AI ..."
[31]: https://en.wikipedia.org/wiki/Kimi_%28chatbot%29 "Kimi (chatbot)"
