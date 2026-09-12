# Repetición espaciada (*spaced repetition*)

Prompts del **capítulo 6**, sección «Técnicas cognitivas potenciadas por IA».

Repasar en intervalos crecientes, apoyándose en la curva del olvido de Ebbinghaus: se repasa justo antes de olvidar, que es cuando el repaso rinde más. El metaanálisis de Cepeda et al. (2006), que reúne 839 comparaciones experimentales, confirma el efecto y muestra que **el intervalo óptimo depende de cuánto tiempo necesites retener el material**: no es lo mismo estudiar para un parcial en dos semanas que para un examen de grado en seis meses.

Junto con la recuperación activa, es la otra técnica que Dunlosky et al. (2013) califican de utilidad alta.

---

## 1. Calendario de repaso base

Punto de partida para un tema visto hoy.

| Sesión | Intervalo | Actividad con IA |
|---|---|---|
| 1.ª revisión | Mismo día (4-6 h) | Quiz corto de 5 preguntas |
| 2.ª revisión | Día siguiente | Test de 10 preguntas con explicaciones |
| 3.ª revisión | 3 días después | Casos prácticos simples |
| 4.ª revisión | 1 semana | Problemas de aplicación |
| 5.ª revisión | 2 semanas | Síntesis y conexiones |
| 6.ª revisión | 1 mes | Evaluación completa simulada |

---

## 2. Generar tu propio calendario

```text
Estoy estudiando [ASIGNATURA]. Tengo [EXAMEN / ENTREGA] el [FECHA] y hoy es
[FECHA DE HOY]. Los temas son:

- [TEMA 1]
- [TEMA 2]
- [TEMA 3]

Diseña un calendario de repaso espaciado hasta la fecha del examen, con
intervalos crecientes. Para cada sesión indica: qué tema toca, qué tipo de
actividad (quiz corto, casos prácticos, simulacro completo) y cuánto tiempo
debería durar.

Dispongo de [N] minutos al día. Devuélvelo como tabla.
```

**Ajusta el horizonte, no solo los temas.** Si el examen es en seis meses, los intervalos finales deben ser mucho más largos que los de esta tabla. Díselo explícitamente.

---

## 3. Ajustar el calendario a tus resultados reales

Este es el prompt que hace que el calendario sea tuyo y no genérico.

```text
Estos son mis resultados de los últimos repasos de [ASIGNATURA]:

- [TEMA 1]: acerté [X] de [Y]
- [TEMA 2]: acerté [X] de [Y]
- [TEMA 3]: acerté [X] de [Y]

Reajusta mi calendario de repaso: acorta los intervalos de los temas donde
fallo más y alárgalos donde ya voy bien. Explícame por qué cambiaste cada
uno.
```

**Por qué importa.** Un estudiante que retiene bien los procedimientos matemáticos pero olvida rápido las fechas necesita dos calendarios distintos dentro de la misma asignatura. Llevar eso a mano, tema por tema, no lo hace nadie; es exactamente el trabajo que conviene delegar.

---

## 4. Repaso relámpago

Para el día anterior, o para los diez minutos muertos entre dos clases.

```text
Hazme 20 preguntas rápidas de los conceptos más importantes de [TEMAS].

Una a la vez, respuesta corta, y dime si acerté antes de pasar a la
siguiente.
```

---

## 5. Herramientas que automatizan esto

- **Anki** — repetición espaciada con un algoritmo maduro y una comunidad amplia de mazos compartidos. Las *flashcards* del archivo [03-recuperacion-activa.md](./03-recuperacion-activa.md) se pueden importar en CSV.
- **RemNote** — combina apuntes y tarjetas en un mismo documento.
- **El calendario de tu teléfono** — menos sofisticado y suficiente. Lo que importa es que el recordatorio traiga ya la primera pregunta dentro, para que empezar el repaso cueste un toque en la pantalla y no una decisión.
