# Guía de Exposición — Conectividad Efectiva Cerebral

## Causal Discovery en Modelos Causales No-Estacionarios

> **Uso durante la presentación:** Este documento está ordenado slide por slide.
> Cada sección tiene: qué decir al abrir el slide, qué señalar en la figura, cómo cerrar y pasar al siguiente.
>
> - Duración objetivo: **25–30 min** de presentación + 10 min de preguntas
> - Ritmo: ~1.5–2 min por slide; la sección del detector (slides 6–12) puede ir a 2 min/slide

---

## Slide 01 — Título

**Lo que dice el slide:**
*"Causal Discovery in Non-Stationary Causal Graphical Models — Applications to Brain Effective Connectivity.
Progress report: Gatekeeper v1-F · CausalMorph integration."*

**Mientras la gente se sienta, puedes decir informalmente:**

> "Buenas [tardes / días]. Voy a hablar de un problema que aparece cuando intentas inferir conexiones causales en el cerebro: el cerebro no se queda quieto. Sus conexiones cambian mientras el sujeto aprende, se recupera de una lesión, o simplemente envejece. Hoy les muestro la infraestructura que estoy construyendo para detectar y recuperar esas conexiones cambiantes a partir de señales hemodinámicas."

---

## Slide 02 — The Brain Changes — Causally

**Apertura:**

> "Comencemos con la motivación. ¿Por qué el cerebro? ¿Y por qué la causalidad?"

**Puntos a desarrollar, en orden:**

1. **Conectividad efectiva** — No es correlación. Es la *influencia causal dirigida* que una región ejerce sobre otra. Si el área prefrontal activa el área motora, existe una arista directed de prefrontal → motor.
2. **Plasticidad** — El cerebro remodela esas conexiones continuamente. Durante una sesión de entrenamiento motor, los circuitos prefrontales-motores se fortalecen progresivamente. Después de un ACV, circuitos alternativos se activan para compensar. En el envejecimiento, conexiones se pierden gradualmente.
3. **fNIRS** — La tecnología que medimos. Infrarrojo cercano sobre el cuero cabelludo, mide la respuesta hemodinámica (oxigenación) como *proxy* de actividad neuronal. Cada canal corresponde a una región cortical; la serie de tiempo es el proxy de esa región. *(Señala la figura: "Estas son nuestras señales — cada color de fondo es un régimen de conectividad diferente.")*
4. **No-estacionariedad** — Las propiedades estadísticas de las señales cambian en el tiempo — esto es inherente, no un artefacto. Kim (2010) lo documenta empíricamente en señales hemodinámicas.
5. **El problema con los métodos clásicos** — LiNGAM, PCMCI, PC asumen un único grafo fijo durante toda la sesión. Si la conectividad cambia, estiman un *promedio* que no describe ningún régimen real.

**Señalar en la figura `fig_10_timeseries.png`:**

- Las **bandas de color de fondo**: *"Cada color es un régimen — una estructura causal diferente. Aquí hay cinco."*
- Las **líneas rojas discontinuas verticales**: *"Estas son las transiciones estructurales — el momento en que el grafo cambia. No las conocemos de antemano."*
- Los **cambios de amplitud visibles entre bandas**: *"La dependencia estadística entre las variables cambia. No es ruido; es una reorganización causal."*

**Cierre / transición:**

> "El desafío central es doble: primero hay que *detectar* cuándo ocurre el cambio, y luego *recuperar* qué grafo aplica en cada intervalo — todo esto a partir de las señales observadas, sin saber de antemano los tiempos de cambio. Eso es lo que hace el sistema que voy a presentar."

---

## Slide 03 — A Three-Pillar Framework

**Apertura:**

> "Antes de entrar en el algoritmo, les sitúo en el marco general de mi doctorado para que tengan contexto sobre dónde estamos hoy."

**Puntos a desarrollar:**

1. **Objetivo general del doctorado** — Desarrollar y validar un framework de causal discovery para redes causales Bayesianas dinámicas no-estacionarias (nsDCBNs), aplicado a señales hemodinámicas cerebrales.
2. **Los tres pilares** — Léelos de la tabla:
  - **Preconditioning (CausalMorph):** Transforma los datos para que LiNGAM pueda identificar el orden causal de forma más robusta, incluso cuando el grafo anterior era parcialmente diferente. Objetivo O4.
  - **Detección (Gatekeeper v1-F):** Detector de cambios estructurales basado en wavelets y momentos estadísticos múltiples. Objetivos O1 y O3. *Este es el núcleo de hoy.*
  - **Aprendizaje (nsDCBN):** Modelo causal que se adapta a través de los regímenes detectados. Parcialmente implementado con LiNGAM por ventana.
3. **Lo que queda pendiente** — O2 (simulador fNIRS), O6 (validación sistemática), O7 (datos reales).
4. **Alcance de hoy** — *"Hoy cubro O1, O3, y O4/O5. El hilo conductor: construí un generador sintético, construí el detector, integré CausalMorph, y evalué todo junto."*

**Cierre / transición:**

> "Empecemos por el problema formal para que los algoritmos tengan contexto matemático claro."

---

## Slide 04 — Formal Problem Statement

**Apertura:**

> "Formalicemos. ¿Qué queremos resolver exactamente?"

**Puntos a desarrollar:**

1. **La serie de tiempo observada** — X ∈ ℝ^{T×p}: T muestras temporales, p canales (regiones cerebrales). No conocemos los tiempos de cambio.
2. **El modelo estructural por régimen** — Cada régimen *r* tiene su propio SCM (Structural Causal Model):
  - X_i(t) = Σ w_ji · X_j(t) + ε_i(t)
  - Los padres causales PA_i^(r) pueden cambiar entre regímenes.
  - Los pesos w_ji^(r) también pueden cambiar.
  - El ruido ε_i^(r) es no-Gaussiano e i.i.d. — esto es crucial para la identificabilidad de LiNGAM (Shimizu 2006).
3. **Los dos objetivos simultáneos:**
  - Detectar los tiempos de cambio {τ̂_k} sin conocerlos a priori.
  - Recuperar el grafo Ĝ_r para cada intervalo detectado.

**Señalar en la figura `fig_14_pipeline_architecture.png`:**

- Las **dos etapas del pipeline** (detección → recuperación): *"Primero segmentamos la serie; luego, en cada segmento, aplicamos el estimador causal."*
- Las **anotaciones de ventana** en la figura: *"Cada intervalo detectado recibe su propio grafo independiente."*

**Cierre / transición:**

> "Para probar el sistema necesitamos datos controlados donde conocemos la verdad. Ese es el Objetivo O1 — el generador sintético."

---

## Slide 05 — O1: Synthetic Learning Trajectory

**Apertura:**

> "Antes de poder evaluar un detector, necesitas datos donde sabes exactamente cuándo ocurren los cambios. Por eso construí un generador sintético inspirado en el aprendizaje motor."

**Puntos a desarrollar:**

1. **Configuración** — p = 5 variables (regiones), R = 5 regímenes, 600–800 muestras por régimen (~3.500 en total). Es un escenario desafiante pero manejable para comenzar.
2. **La trayectoria de aprendizaje** — El grafo evoluciona de G_init (disperso) a G_target (más denso). Modela cómo la conectividad se fortalece y reorganiza durante el aprendizaje de una habilidad motora.
3. **Dos fuentes de no-estacionariedad:**
  - **Topológica:** aristas que aparecen o desaparecen entre regímenes.
  - **Paramétrica:** los pesos y la escala del ruido se remuestrean en cada régimen, incluso cuando la topología se mantiene igual.
4. **Por qué importa la segunda fuente** — Un detector que solo busca cambios topológicos fallaría cuando el grafo es el mismo pero los pesos cambian drásticamente. El nuestro captura ambos.

**Señalar en la figura `fig_11_structures_comparison.png` (fila superior únicamente):**

- **Aristas que aparecen y desaparecen** de izquierda a derecha: *"En cada panel, el cerebro tiene una configuración causal diferente."*
- **Posiciones fijas de los nodos**: *"Las mismas cinco regiones cerebrales — lo que cambia es cómo se conectan."*
- **Progresión de disperso a denso**: *"Esto modela la consolidación de la memoria motora: más conexiones se establecen con la práctica."*

**Cierre / transición:**

> "Tenemos los datos sintéticos con la verdad conocida. Ahora veamos el detector — siete pasos, cada uno con una razón específica."

---

## Slide 06 — Gatekeeper v1-F: Seven-Step Pipeline

**Apertura:**

> "El Gatekeeper v1-F es el núcleo algorítmico de esta presentación. Es un detector de cambios estructurales en series de tiempo multivariadas. Tiene siete pasos. Cada paso resuelve un problema específico que observé en versiones anteriores. Les doy el mapa primero; luego entramos en los pasos interesantes uno por uno."

**Recorre la tabla en voz alta:**


| Paso | Qué hace                                           | Por qué importa                                                |
| ---- | -------------------------------------------------- | -------------------------------------------------------------- |
| 0    | Rechazo de artefactos (Mediana + MAD)              | Elimina outliers antes de cualquier análisis                   |
| 1    | Momentos rolling (media, var, asimetría, curtosis) | Captura cambios distribucionales, no solo de media             |
| 2    | CWT Morlet multi-escala                            | Localiza cambios en tiempo Y frecuencia simultáneamente        |
| 3    | Calibración por surrogados de Fourier              | Umbrales controlados bajo la hipótesis nula de estacionariedad |
| 4    | Z = log(𝒲/θ) — signed log-ratio                   | Detecta tanto subidas como bajadas de conectividad             |
| 5    | Derivada dE + find_peaks                           | Localiza *transiciones*, no estados sostenidos                 |
| 6    | Validación de paso + gate K-de-canales (v1-F)      | Rechazo de falsos positivos en dos etapas                      |


> "Los pasos 4 y 6 son las innovaciones clave respecto a versiones anteriores. El paso 4 agrega los cuatro momentos con pesos; el paso 6 añade la consistencia multi-canal. Vamos paso a paso."

---

## Slide 07 — Step 1: Rolling Moments

**Apertura:**

> "¿Por qué cuatro momentos? La intuición es simple: un cambio estructural puede manifestarse en *cualquier* momento estadístico, no solo en la media."

**Puntos a desarrollar:**

1. **Ejemplo de conectividad cerebral** — Cuando se agrega una conexión efectiva entre dos regiones, la varianza del receptor *siempre* aumenta porque ahora recibe input adicional. La media puede no moverse si el nuevo input tiene media cero.
2. **Ejemplo de fNIRS** — La asimetría (skewness) de la respuesta hemodinámica cambia en el onset de una tarea *antes* de que la media lo haga, porque la rampa de subida es más abrupta que la bajada.
3. **La tabla de cuándo responde cada momento** — La varianza responde siempre que se agrega una arista; la curtosis responde cuando la distribución del ruido cambia de forma; la asimetría responde ante cambios en la distribución del noise no-Gaussiano.

**Señalar en la figura `fig_02_rolling_moments.png` (de arriba a abajo):**

- **Panel 1 (señal cruda)**: *"Aquí ocurre el cambio en t=350. Visualmente parece solo un aumento de amplitud."*
- **Panel 2 (M1, media, rojo)**: *"La media apenas se mueve. Un detector clásico basado en CUSUM de media sze lo perdería por completo."*
- **Panel 3 (M2, varianza, azul)**: *"La varianza se duplica bruscamente. Esta es la señal dominante del cambio estructural."*
- **Paneles 4–5 (M3/M4)**: *"Asimetría y curtosis también reaccionan, con menor amplitud. Son señales secundarias pero útiles."*
- **Línea roja discontinua vertical en t=350**: *"El cambio real — todos los paneles lo marcan."*

**Cierre / transición:**

> "Ahora tenemos cuatro señales de cambio por canal. El siguiente paso es transformarlas en el dominio tiempo-frecuencia para poder localizar con precisión cuándo ocurre el cambio."

---

## Slide 08 — Step 2: CWT — Multi-Scale Localisation

**Apertura:**

> "¿Por qué wavelets? La alternativa natural es la STFT — transformada de Fourier de ventana corta. El problema es que la STFT tiene una ventana fija: si usas una ventana larga, tienes buena resolución en frecuencia pero mala en tiempo; si usas una ventana corta, al revés. Las wavelets resuelven esto adaptativamente."

**Puntos a desarrollar:**

1. **La CWT Morlet** — Analiza la señal a múltiples escalas simultáneamente. A escalas pequeñas (alta frecuencia) la ventana es corta y tiene buena resolución temporal. A escalas grandes (baja frecuencia) la ventana es larga y tiene buena resolución espectral. Exactamente lo que necesitamos para señales hemodinámicas que tienen energía en múltiples bandas de frecuencia.
2. **La firma visual de un cambio estructural** — Un cambio real genera una **banda vertical brillante** que se extiende por *todas* las escalas a la misma posición temporal. Un artefacto de ruido o sensor genera energía dispersa en una o dos escalas aisladas.
3. **El parámetro ω₀ = 6** — Compatible con las frecuencias de la señal hemodinámica (0.01–0.3 Hz). No es un parámetro libre; está justificado por el dominio.

**Señalar en la figura `fig_03_cwt_scalogram.png` (de arriba a abajo):**

- **Panel superior (señal cruda)**: *"Cambio de varianza en t=400."*
- **Panel medio (varianza rolling)**: *"M² se duplica en el punto de cambio — confirmación en dominio tiempo."*
- **Panel inferior (escalograma CWT)**: *"Esta es la figura clave. La banda vertical brillante que atraviesa TODAS las escalas en t=400 es la firma de un cambio estructural genuino."*
- **La región oscura antes del cambio**: *"Energía uniformemente baja bajo estacionariedad."*
- Contraste con ruido: *"El ruido produciría píxeles brillantes dispersos, no una banda. Esta consistencia multi-escala es lo que distingue el cambio real."*

**Cierre / transición:**

> "Ahora tenemos el mapa de energía en tiempo-escala. Pero ¿cómo decidimos si la energía en un punto dado es 'demasiado grande'? Necesitamos un umbral que esté calibrado en la señal específica que estamos analizando."

---

## Slide 09 — Step 3: Fourier Surrogate Calibration

**Apertura:**

> "Este es el paso estadístico del pipeline. La pregunta es: ¿cómo definimos 'energía anormalmente alta' sin asumir una distribución paramétrica para la señal cerebral?"

**Puntos a desarrollar:**

1. **El problema de un umbral fijo** — Las señales fNIRS tienen ruido 1/f (ruido de tipo flicker): la energía a frecuencias bajas es naturalmente más alta que a frecuencias altas. Un umbral plano generaría alarmas falsas constantes en escalas grandes.
2. **La solución: surrogados de Fourier** — Generamos K = 100 señales sintéticas que:
  - Preservan exactamente el **espectro de potencia** de la señal original (misma autocorrelación).
  - Son estacionarias por construcción — sin cambios de régimen.
  - La fase de cada componente de Fourier se aleatoriea con uniforme(0, 2π).
3. **El umbral por escala** — θ_s = cuantil_{1−α} de la distribución de energía de los surrogados en esa escala. Un umbral diferente para cada escala: las escalas de alta frecuencia tienen umbrales más altos porque el ruido natural allí es mayor.
4. **α = 0.40** — Permisivo, porque queremos sensibilidad en escenarios de múltiples regímenes. Un α estricto (0.05) perdería cambios de conectividad débiles. En datos reales se calibraría en la línea base de reposo del participante.

**Señalar en la figura `fig_04_surrogate_calibration.png`:**

- **Panel izquierdo**: *"Línea oscura = señal de línea base original. Colores = tres surrogados. Misma envolvente de amplitud, distinta fluctuación temporal — estacionarios por construcción."*
- **Panel derecho (escalones)**: *"Un valor de umbral por cada escala de CWT. Las escalas pequeñas tienen umbrales más altos porque hay más ruido a cortos plazos temporales."*

**Cierre / transición:**

> "Con los umbrales calibrados, podemos convertir el escalograma en una señal de desviación firmada — que nos diga no solo cuándo hay un cambio, sino también si es una *subida* o una *bajada* de conectividad."

---

## Slide 10 — Step 4: Signed Log-Ratio + Moment Aggregation

**Apertura:**

> "Este es el paso de integración. Combinamos los cuatro momentos con los cuatro canales y todas las escalas en una única señal de energía firmada."

**Puntos a desarrollar:**

1. **El log-ratio firmado** — Z = log(𝒲/θ):
  - Z > 0: la energía supera la línea base → **subida** (nueva conexión, mayor peso).
  - Z < 0: la energía cae por debajo de la línea base → **bajada** (conexión debilitada).
  - La simetría es crítica: en un paradigma de aprendizaje, algunas conexiones se fortalecen mientras otras se debilitan simultáneamente. Un detector unidireccional perdería la mitad de los cambios.
2. **Los pesos factoriales inversos** — w_m = 1/m!:
  - Media (m=1): w = 1.0 — el momento más estable estadísticamente.
  - Varianza (m=2): w = 0.5 — muy informativa para cambios causales.
  - Asimetría (m=3): w ≈ 0.17 — señal secundaria.
  - Curtosis (m=4): w ≈ 0.04 — mínima confianza, máxima varianza estadística.
  - **Por qué no aprender los pesos:** requeriría datos etiquetados con tiempos de cambio conocidos, lo que anula el propósito de un detector no supervisado.
3. **La agregación** — Sumamos la diferencia entre energía positiva y negativa, pesada por momento, a través de todos los canales y escalas. El resultado es E_signed(t): un escalar por cada instante.

**Señalar en la figura `fig_06_multimoment_aggregation.png`:**

- **Gráfico de barras (arriba izquierda)**: *"Los pesos factoriales — media = mayor confianza, curtosis = menor."*
- **Arriba derecha**: *"Las energías de los cuatro momentos en el punto de cambio — varianza (azul) lidera."*
- **Paneles medios/inferiores**: *"Energía firmada de cada momento individualmente — todos tienen un pico cerca de t=350."*
- **Por qué la combinación ayuda**: *"Si la media no reacciona pero la varianza sí, la señal combinada igual dispara."*

**Cierre / transición:**

> "Tenemos E_signed(t). Ahora hay un problema sutil con umbralar esta señal directamente."

---

## Slide 11 — Step 5: Derivative Detection

**Apertura:**

> "Aquí viene una decisión de diseño que fue clave para que el detector funcionara en la práctica."

**Puntos a desarrollar:**

1. **El problema con umbralar E_signed directamente** — Después de un cambio estructural, la conectividad se *mantiene* elevada en el nuevo régimen. Esto significa que E_signed también se mantiene elevada durante todo el nuevo régimen. Si umbralizamos directamente, el detector dispara en *cada muestra* del nuevo régimen — cientos de falsas alarmas por cada cambio real.
2. **La solución: diferenciar primero, luego detectar picos** — ΔE(t) = E_signed(t) − E_signed(t−1), con suavizado (ventana 12 muestras).
  - ΔE es grande solo **en la transición misma**, no durante el régimen.
  - Pico positivo → inicio de un régimen de mayor conectividad.
  - Pico negativo → inicio de un régimen de menor conectividad.
3. **Detección de picos** — `scipy.signal.find_peaks`:
  - Altura mínima ε calibrada en los surrogados (altura típica de picos de ruido).
  - Período refractario = 150 muestras — evita múltiples detecciones por la misma transición.
4. **Analogía en neurociencia** — Los detectores de eventos electrofisiológicos (detección de spikes en EEG) operan con la misma lógica: diferencian antes de umbralizar para localizar el inicio del evento.

**Señalar en la figura `fig_07_derivative_detection.png` (tres paneles):**

- **Panel superior (E_signed)**: *"La energía se mantiene alta después del cambio — umbralizar aquí dispara en cada muestra del nuevo régimen."*
- **Panel medio (ΔE)**: *"La derivada es grande SOLO en la transición — localización temporal perfecta."*
- **Líneas de umbral punteadas**: *"Calibradas a partir de alturas de pico en los surrogados — menos conservadoras que la estadística de máximo."*
- **Panel inferior**: *"find_peaks identifica el pico; triángulo rojo = cambio de punto detectado. Coincide con la línea verde de verdad conocida."*

**Cierre / transición:**

> "Ahora tenemos candidatos a cambios de punto. El último paso es filtrar los que podrían ser artefactos de un solo canal."

---

## Slide 12 — Step 6b: K-of-Channels Gate (v1-F)

**Apertura:**

> "Esta es la innovación específica de la versión v1-F — el sufijo F viene de 'fNIRS-aware'. Es un filtro diseñado explícitamente para los artefactos más comunes en fNIRS."

**Motivación fNIRS:**

Los tres artefactos más frecuentes en fNIRS afectan canales *individuales*:

- **Artefacto de movimiento**: el sujeto mueve la cabeza → un optodo se desacopla.
- **Ruido de acoplamiento de optodo**: el optodo no tiene buen contacto con el cuero cabelludo.
- **Oclusión por cabello**: el cabello bloquea un canal específico.

Un cambio de conectividad *real* debe ser **distribuido**: si el área prefrontal cambia su patrón de disparo, eso se refleja en *múltiples* canales simultáneamente.

**El gate en detalle:**

1. Para cada candidato τ, calcula el z-score de paso por canal: z_n(τ) = (mediana(E_n[post]) − mediana(E_n[pre])) / σ̂_n.
2. **Acepta el candidato si y solo si ≥ k_ch = 2 canales tienen |z_n| ≥ 1.5** — al menos dos regiones cerebrales muestran cambio significativo.
3. También reporta el **ratio de concentración** R(τ) = max|Δ_n| / Σ|Δ_n|:
  - R ≈ 1/N: cambio distribuido uniformemente → cambio genuino de conectividad.
  - R ≈ 1: prácticamente todo el cambio está en un solo canal → artefacto.

**Señalar en la figura `fig_08_channel_gate.png`:**

- **Panel izquierdo**: *"Canales verdes se elevan claramente; canales grises son planos — ruido o artefacto. Tres de cinco canales son activos — este candidato PASA el gate."*
- **Panel derecho (barras de z-scores)**: *"La línea discontinua es el umbral k_z = 1.5. Tres barras lo superan."*
- **Ratio de concentración en el subtítulo**: *"R = 0.31, cercano a 1/5 = 0.20 — cambio distribuido, no concentrado en un canal."*

**Nota técnica si preguntan:**
El gate se deshabilita automáticamente para N = 1 canal y se ajusta a max(1, N//2) para N pequeño. En dispositivos fNIRS reales con 16–64 canales, opera a plena potencia.

**Cierre / transición:**

> "Con los siete pasos completos, vamos a ver el resultado en nuestro escenario de cinco regímenes."

---

## Slide 13 — Full Detection Result (O3)

**Apertura:**

> "Esta figura muestra el resultado completo del Gatekeeper v1-F en la señal de cinco regímenes. Hay ocho paneles — vamos de arriba a abajo."

**Guía panel por panel (señala mientras hablas):**

1. **Panel 1 (señales originales)** — *"Cinco variables. Las líneas verdes discontinuas son los cambios de punto detectados; las líneas rojas sólidas son la verdad conocida."*
2. **Panel 2 (ΔE, la señal de detección)** — *"Los picos se alinean con las transiciones de régimen. Pueden ver que la señal es selectiva — no dispara en cada muestra, solo en las transiciones."*
3. **Panel 3 (E_signed)** — *"La energía firmada agregada. Niveles claramente escalonados en cada transición — la señal tiene estructura."*
4. **Paneles 4–7 (por momento)** — *"La varianza (azul) domina. La asimetría y la curtosis añaden evidencia secundaria. Sin los momentos superiores, el detector sería menos sensible a cambios de distribución del ruido."*
5. **Panel 8 (severidad por leaky integrator)** — *"Evidencia acumulada de no-estacionariedad. Un indicador de cuán lejos está el sistema del comportamiento estacionario."*

**Resultado de detección:**

> "En este experimento, el detector captura 3 de 4 puntos de cambio — aproximadamente 75%. El punto que falla es una transición débil donde solo se intercambia una arista. Volveré a esto en el slide de resultados."
>
> "Con los tiempos de cambio correctos (oráculo), el sistema reconstruye la causalidad — lo vemos ahora."

---

## Slide 14 — O4: CausalMorph Integration

**Apertura:**

> "Con los segmentos detectados, necesitamos recuperar el grafo causal en cada ventana. Aquí entra CausalMorph."

**El problema con DirectLiNGAM en frío:**

> "Si aplicamos DirectLiNGAM a cada ventana sin contexto, obtenemos SHD ≈ 0.5 y F1 ≈ 0.0 — prácticamente sin poder recuperar el grafo. Hay dos razones: las ventanas son cortas (600–800 muestras para p=5) y la señal aún tiene mezcla de la conectividad anterior."

**La solución CausalMorph:**

X' = (I − B̂^(r-1)) · X

Esta transformación resta la contribución predicha de los padres causales del régimen anterior. El resultado X' se aproxima a los residuos del ruido del nuevo régimen — más cerca de ser i.i.d. y no-Gaussianos, que es la condición que LiNGAM necesita para identificar el orden causal.

**La cadena warm-start:**

- Ventana 0: usamos el grafo del régimen 0 como prior (en un experimento real, este vendría de un scan de reposo al inicio de la sesión).
- Ventana k ≥ 1: el prior es la salida de DirectLiNGAM de la ventana anterior.
- La cadena propaga el conocimiento estructural acumulado a medida que avanzan los regímenes.

**Señalar en la figura `fig_11_structures_comparison.png` (ambas filas):**

- **Fila superior (azul — grafos verdaderos)**: *"La verdad conocida — cinco regímenes, cinco grafos diferentes."*
- **Fila inferior (verde — grafos aprendidos)**: *"Lo que recupera CausalMorph + LiNGAM."*
- **Etiquetas de aristas (coeficientes)**: *"Signo y magnitud de las conexiones estimadas."*
- **Valores de nSHD en subtítulos**: *"SHD normalizado — 0 es perfecto, 1 es el peor caso. El régimen 0 es perfecto porque el prior viene de la verdad conocida."*

**Cierre / transición:**

> "¿Qué tan bien funciona cuantitativamente? Eso es O5."

---

## Slide 15 — O5: Performance Evaluation

**Apertura:**

> "Los resultados cuantitativos. Hay dos métricas: detección de cambios de punto y recuperación de estructura causal."

**Detección de cambios de punto:**

- Tolerancia de ±125 muestras para considerar una detección correcta.
- **Oráculo** (verdad conocida dada directamente a CausalMorph): 3/3 detectados — 100%. Techo de rendimiento.
- **Gatekeeper v1-F** (detector real): ~75% en experimentos anteriores.

**Recuperación causal con segmentación oráculo:**

*(Léela de la tabla, régimen por régimen)*


| Régimen   | nSHD      | Interpretación                                                    |
| --------- | --------- | ----------------------------------------------------------------- |
| 0         | 0.000     | Perfecto — prior del régimen 0 = verdad conocida                  |
| 1         | 0.167     | Una arista incorrecta                                             |
| 2         | 0.333     | Peor: 3 aristas activas, prior del régimen 1 con error propagado  |
| 3         | 0.083     | Muy bueno — cadena se estabiliza con más datos acumulados         |
| **Media** | **0.146** | Bajo segmentación oráculo, el paso causal es el cuello de botella |


**Señalar en la figura `fig_13_shd_metrics.png`:**

- **Barras de SHD por régimen**: *"Menor es mejor. El régimen 0 es perfecto."*
- **Régimen 2 más alto**: *"La ventana más compleja — tres aristas activas y el prior viene del régimen 1 que ya tenía error."*
- **Régimen 3 se recupera**: *"A pesar de ser el régimen más denso, la cadena warm-start se estabiliza. El sistema aprende a manejar las correcciones."*
- **Media = 0.146**: *"Bajo segmentación perfecta, el paso de recuperación causal es el cuello de botella. Eso guía nuestro trabajo futuro en O3 y O6."*

**Punto clave:**

> "Estos resultados usan la segmentación oráculo — los tiempos de cambio verdaderos se alimentan directamente a CausalMorph. Esto *aísla* la calidad de la recuperación causal del error del detector. La conclusión es que, incluso con segmentación perfecta, el régimen 2 es difícil por la propagación de error en la cadena."

---

## Slide 16 — Summary and Next Steps

**Apertura:**

> "Resumo lo que construí, lo que funciona, y lo que viene después."

**Lo que se construyó (leer de la lista):**

- **O1 ✓** — Motor de generación de DAGs multi-régimen con verdad controlada. Trayectoria de aprendizaje cerebral sintético.
- **O3 ✓** — Gatekeeper v1-F: detector multi-momento con calibración por surrogados, energía firmada, derivada de picos, y gate de consistencia K-de-canales.
- **O4 ✓ (parcial)** — CausalMorph + DirectLiNGAM con warm-start iterativo por ventana detectada.
- **O5 ✓** — Evaluación sobre datos sintéticos: nSHD por régimen + porcentaje de cambios de punto detectados.

**Limitaciones abiertas (honestas):**

- Detector: ~75% de CP detectados; las transiciones débiles de una sola arista se pierden a veces.
- CausalMorph con segmentación oráculo: nSHD media = 0.146; régimen 2 llega a 0.333 por propagación de error en el prior.
- La cadena warm-start es sensible a una ventana mal estimada.
- Solo se testaron transiciones abruptas — transiciones graduales son trabajo futuro.

**Roadmap del doctorado:**

- **O2** — Simulador fNIRS neuro-dinámico: función de respuesta hemodinámica realista, modelo de ruido específico del dominio.
- **O3 continúa** — Mejorar sensibilidad del detector para cambios débiles; probar en transiciones graduales.
- **O6** — Validación sistemática: niveles de ruido, intensidad de no-estacionariedad, densidad de aristas.
- **O7** — Datos reales de fNIRS: evaluación observacional de conectividad efectiva cerebral.

**Cierre final:**

> "El pipeline funciona de extremo a extremo en datos sintéticos. El próximo hito inmediato es O2 — construir un simulador fNIRS específico para testear en señales que igualen las características espectrales y de ruido de los datos hemodinámicos reales. Ese simulador también nos permitirá validar la calibración por surrogados, que actualmente usa un modelo genérico de Fourier.
>
> Gracias. Con gusto respondo preguntas."

---

## Preguntas Anticipadas — Respuestas en Español

### P1: ¿Cómo se relaciona esto con CD-NOD y otros métodos no-estacionarios?

> "CD-NOD detecta no-estacionariedad probando si los residuos de un modelo causal dependen de una variable auxiliar de tiempo. Nuestro enfoque es complementario: primero segmentamos la serie temporal usando energía wavelet, luego ajustamos un modelo causal separado por segmento. CD-NOD detecta la *existencia* de no-estacionariedad; nosotros además la *localizamos* y *recuperamos la estructura por régimen*. Una comparación cuantitativa en benchmarks compartidos es un ítem pendiente importante que cae en O6."

---

### P2: ¿Por qué fNIRS y no fMRI o EEG?

> "fNIRS es portátil, tolera movimiento, y es más barato — lo que lo hace adecuado para paradigmas de aprendizaje naturalistas: registrar a un sujeto practicando una habilidad motora en un escritorio. La señal hemodinámica es más lenta que EEG pero tiene mejor resolución espacial que el EEG de cuero cabelludo, y produce naturalmente las señales multi-canal de cambio de régimen que nuestro framework apunta. En O7 usaremos datos reales de fNIRS."

---

### P3: ¿Por qué LiNGAM y no un método Bayesiano o basado en score?

> "LiNGAM explota la no-Gaussianidad para identificar un grafo causal único sin necesitar búsqueda sobre el espacio de DAGs. Es O(p³) por ventana y da un estimador puntual — suficientemente rápido para reajustar en cada punto de cambio detectado. Los métodos basados en score como GES o NOTEARS son más flexibles pero más lentos y requieren un argumento de identificabilidad diferente. LiNGAM encaja bien con nuestro modelo de ruido no-Gaussiano i.i.d."

---

### P4: ¿Qué tan sensible es el detector a α = 0.40?

> "α controla el cuantil del surrogado usado para el umbral. Un α menor es más estricto: reduce falsos positivos pero también pierde cambios más débiles. 0.40 fue ajustado para el escenario multi-régimen sintético. Para datos reales de fNIRS, α debería calibrarse en una línea base de reposo conocida como estacionaria de cada participante — eso está contemplado en O6."

---

### P5: ¿Qué está haciendo geométricamente la transformación CausalMorph?

> "X' = (I − B̂)X resta la contribución predicha de los padres de cada nodo de su propia señal. En el caso ideal, X' ≈ residuos de ruido: i.i.d., no-Gaussianos, media cero. LiNGAM opera mejor sobre residuos, así que este pre-blanqueamiento facilita la recuperación del orden causal incluso cuando el nuevo régimen difiere solo parcialmente del anterior."

---

### P6: ¿Por qué el detector pierde el primer punto de cambio?

> "La línea base es el primer 50% del régimen 0. La primera transición ocurre al final del régimen 0, cerca del límite de la ventana de línea base. Los umbrales surrogados se calibran en la *línea base completa*, así que los aumentos locales de energía cerca del límite tienen menor poder estadístico. Una línea base de reposo pre-sesión de calibración — estándar en protocolos de fNIRS — resolvería esto. Está contemplado como mejora de O3 en O7."

---

### P7: ¿El gate K-de-canales falla para pocos canales?

> "El gate se deshabilita automáticamente para N = 1 (caso de un solo canal) y se fija a max(1, N//2) para N pequeño. Para N = 2 el umbral efectivo es 1 canal — equivalente a no tener gate. En la práctica, los dispositivos fNIRS tienen 16–64 canales en las regiones de interés, así que el gate opera a plena potencia. Está contemplado en O7."

---

### P8: ¿Por qué pesos factoriales inversos en lugar de pesos aprendidos?

> "Son una elección no-paramétrica con principio: los momentos más bajos son estadísticamente más estables con tamaños de ventana pequeños, así que naturalmente merecen más peso. Aprender los pesos requeriría datos etiquetados con tiempos de cambio de régimen conocidos, lo que anula el propósito de un detector no supervisado. El esquema 1/m! también es interpretable y reproducible entre datasets."

---

### P9: ¿Qué representa el prior de Ventana-0 en una sesión real de fNIRS?

> "En una aplicación real, el prior de Ventana-0 sería la conectividad estimada de un scan de reposo antes de que comience la tarea — un protocolo estándar en neuroimagen. La cadena warm-start luego sigue cómo la conectividad evoluciona desde esa línea base a través de la sesión. Esto es lo que O7 validará con datos reales."

---

### P10: ¿Qué métrica usarás cuando no haya verdad conocida en datos reales?

> "Sin verdad conocida cambiamos a: (a) confiabilidad test-retest a través de sesiones repetidas, (b) consistencia espacial con la anatomía conocida — ¿aparece la arista prefrontal → motor durante tareas motoras?, y (c) comparación con la conectividad funcional de fMRI como validación indirecta. Este es el foco de O7."

---

*Documento generado para uso del expositor. Versión en español de `presentation_plan.md`.*