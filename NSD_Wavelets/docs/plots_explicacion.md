# Explicación detallada de los plots de presentación

Generados por `presentation_plots.py`.  
Archivo de salida: `slide14_motivation.png` y `slide15_pipeline.png` (180 dpi, fondo blanco).

---

## Plot 1 — `slide14_motivation.png`

**Título del slide:** *"Year 3 — The detector had to change"*  
**Mensaje central:** detectar cambios en la media no era suficiente; los cambios estructurales pueden aparecer en la varianza, asimetría, curtosis o frecuencia.

### Qué hace este plot

Demuestra con un experimento sintético que los tests clásicos de estacionariedad (KPSS, ADF) operan únicamente sobre la media, y por lo tanto son ciegos ante regímenes que difieren en momentos de orden superior. El wavelet approach, al aplicar CWT sobre los momentos rodantes, detecta todos los cambios.

---

### Datos sintéticos generados

La señal tiene **T = 1 000 muestras** divididas en **4 regímenes de 250 muestras cada uno**. La media de todos los regímenes es ≈ 0 por construcción.

| Régimen | t | Distribución | Media | Varianza | Skewness | Kurtosis exceso |
|---------|---|---|---|---|---|---|
| 1 — Baseline | 0–249 | Normal(0, 1) | 0 | 1 | 0 | 0 |
| 2 — Variance jump | 250–499 | Normal(0, 3) | 0 | **9** | 0 | 0 |
| 3 — Positive skew | 500–749 | Exp(1) centrada y normalizada | 0 | 1 | **≈ +2** | ≈ +6 |
| 4 — Heavy tails | 750–999 | Laplace(0, 1/√2) | 0 | 1 | 0 | **≈ +3** |

**Régimen 3 en detalle:** se genera con `rng.exponential(1.0, seg)` y luego se centra en 0 y se normaliza a varianza 1. Esto preserva la forma asimétrica (cola larga a la derecha) con skewness ≈ 2, pero lleva la media y la varianza a los mismos valores que el régimen baseline. El kurtosis exceso también sube (≈ 6), lo que es una consecuencia inevitableque de la asimetría extrema, pero el mensaje en el plot se focaliza en la skewness.

**Régimen 4 en detalle:** `Laplace(0, b)` con `b = 1/√2 ≈ 0.707`. La distribución de Laplace tiene varianza `2b² = 1` (igual que R1) y skewness = 0 (es simétrica), pero sus colas son exponenciales en lugar de gaussianas, lo que produce kurtosis exceso = 3. Visualmente la señal "parece" similar a R1, pero estadísticamente es muy diferente.

---

### Estructura de los 5 paneles

Los colores de fondo (azul claro, naranja claro, verde claro, rosa claro) delimitan los 4 regímenes en **todos los paneles simultáneamente**. Las líneas verticales rojas discontinuas marcan los 3 cambios de punto verdaderos en t = 250, 500, 750.

#### Panel 0 — Señal cruda `x(t)`
- Traza azul (`#2E86AB`), linewidth 0.45, alpha 0.75.
- En el tope de cada banda de régimen hay etiquetas de texto pequeñas que describen la distribución.
- **Qué ver:** a ojo, los regímenes 1 y 4 parecen casi indistinguibles. El régimen 2 es visiblemente más ruidoso. El régimen 3 muestra valores extremos positivos más frecuentes que negativos, pero es sutil.
- **Mensaje implícito:** ningún humano puede ver los 4 cambios sólo mirando la señal cruda.

#### Panel 1 — Rolling Mean (gris-azul `#546E7A`)
- Ventana rodante: 80 muestras, centrada (reflect padding).
- La media rodante oscila cerca de 0 durante **toda** la señal.
- **Anotación central (rojo sobre fondo rosa):** `"KPSS / ADF only tests the mean → sees nothing here"`.
- **Marcadores ✗ rojos** en cada punto de cambio: los tres cambios son invisibles para este detector.
- **Por qué KPSS/ADF falla aquí:** KPSS testa la hipótesis nula de estacionariedad alrededor de una media constante. ADF testa raíz unitaria (que también opera sobre el nivel medio). Ambos tests son agnósticos respecto a varianza, forma o colas.

#### Panel 2 — Rolling Variance (naranja `#F57C00`)
- Varianza rodante (ventana 80).
- **Cambio visible en t = 250:** la varianza salta de ≈ 1 a ≈ 9 cuando entra R2, y vuelve a ≈ 1 en t = 500.
- **Marcadores ✓ verdes** en t = 250 y t = 500: la varianza detecta la entrada y salida de R2.
- El cambio en t = 750 (R4) no produce un salto claro porque Laplace(0, 1/√2) también tiene varianza ≈ 1.

#### Panel 3 — Rolling Skewness (verde `#2E7D32`)
- Skewness rodante (ventana 80, bias=False).
- **Cambio visible en t = 500:** la skewness sube de ≈ 0 a ≈ 2 cuando entra R3 (distribución exponencial centrada).
- **Marcadores ✓ verdes** en t = 500 y t = 750.
- En R1 y R2 la skewness es ≈ 0 porque ambas son normales (simétricas). En R4 (Laplace) también ≈ 0.

#### Panel 4 — Rolling Kurtosis excess (morado `#6A1B9A`)
- Kurtosis exceso rodante (ventana 80, Fisher=True, bias=False).
- Normal tiene kurtosis exceso = 0. Exponencial ≈ 6. Laplace ≈ 3.
- **Cambios visibles** en t = 500 (subida por R3) y t = 750 (bajada parcial porque Laplace tiene kurtosis menor que Exp).
- **Marcadores ✓ verdes** en t = 500 y t = 750.

#### Leyenda flotante (esquina derecha)
Texto sobre fondo blanco: `✗ missed by KPSS / ADF` / `✓ detected by wavelet moment analysis`.

---

### Cómo leer el plot en la presentación

> *"Miren la fila del medio — la media está completamente plana. KPSS y ADF solo ven esa fila. Los tres cambios de régimen son completamente invisibles para ellos. Ahora miren las tres filas de abajo: varianza, skewness, kurtosis — cada una captura exactamente el cambio que le corresponde."*

---

## Plot 2 — `slide15_pipeline.png`

**Título del slide:** *"Gatekeeper v1-F"*  
**Mensaje central:** mostrar cada etapa del pipeline paso a paso sobre la misma señal, de modo que el audiencia vea cómo la información fluye y se transforma hasta llegar a una detección robusta.

### Datos sintéticos generados

Señal multivariada: **T = 800 muestras, N = 4 canales**.

- **Régimen 1** (t = 0–399): `Normal(0, 1)` en todos los canales — baseline.
- **Régimen 2** (t = 400–799): `Normal(0, 2.4)` en todos los canales — **varianza × 5.76**.
- **Artefacto:** en t = 240 (30% de T), canal 0 recibe un spike de +22 unidades.
- El cambio verdadero está en **t = 400** (línea vertical roja discontinua en todos los paneles).

La señal es deliberadamente simple (cambio de varianza puro, todos los canales simultáneamente) para que el pipeline tenga algo claro que detectar, y el artefacto sirve para demostrar la etapa de limpieza.

---

### Estructura de los 6 paneles

Todos los paneles comparten el eje x (0–800 muestras) y muestran la línea roja discontinua en t = 400.

---

#### Panel ① — Raw signal → MAD artifact rejection

**Datos mostrados:**
- Traza gris clara (`#BDBDBD`, lw 0.7): señal cruda del canal 0 con el spike visible.
- Traza azul (`#2E86AB`, lw 1.1): señal limpia después de `remove_gross_artifacts`.
- Marcador rojo ×: posición exacta del artefacto detectado y removido.

**Cómo funciona la limpieza (MAD):**
1. Calcular mediana `m` y `MAD = median(|x - m|)`.
2. Escalar: `MAD_scaled = 1.4826 × MAD` (corrección para distribución normal).
3. Declarar "malo" todo punto donde `|x - m| / MAD_scaled > 5.0`.
4. Interpolar linealmente los puntos malos usando los buenos vecinos.

**Por qué MAD y no z-score:** el z-score usa la media y la desviación estándar, que son sensibles a outliers. Si hay un spike de +22, la media sube y el z-score de otros puntos baja, "enmascarando" al outlier. MAD usa la mediana, que es robusta.

**Qué ver:** la traza azul corre sobre la gris, idénticas salvo en t ≈ 240 donde la azul "cierra" el agujero que dejó el spike.

---

#### Panel ② — Rolling moments (mean · variance · skewness · kurtosis)

**Datos mostrados:**
Las 4 trazas están **normalizadas individualmente** dividiendo por su valor absoluto máximo (`y / max(|y|)`), por lo que todas están en el rango [-1, 1]. Esto permite comparar su dinámica en el mismo eje.

| Traza | Color | Comportamiento esperado |
|-------|-------|---|
| mean | gris-azul `#546E7A` | plana ≈ 0 en ambos regímenes |
| variance | naranja `#F57C00` | **salta en t = 400** de bajo a alto |
| skewness | verde `#2E7D32` | ≈ 0 en ambos (ambas son normales, simétricas) |
| kurtosis | morado `#6A1B9A` | ≈ 0 en ambos (ambas son normales, cola normal) |

**Qué ver:** la varianza naranja es la única que muestra un escalón claro en t = 400. Las otras 3 trazas permanecen planas — lo que implica que el cambio estructural aquí es puramente de varianza. Esto justifica por qué el pipeline computa CWT sobre la varianza rodante en el paso siguiente.

**Ventana:** 60 muestras, centrada con reflect-padding (implementación en `compute_rolling_moments`).

---

#### Panel ③ — CWT Morlet scalogram (sobre varianza rodante)

**Datos mostrados:**
- Imagen 2D (heatmap `inferno`): filas = 32 escalas geométricamente espaciadas entre 2 y T/4 = 200 muestras; columnas = tiempo.
- Intensidad de color = potencia wavelet `|W(scale, t)|²`.
- Línea blanca discontinua en t = 400.

**Cómo se construye:**
1. Se toma la traza de varianza rodante `moms[2]` del canal 0.
2. Se rellanan NaNs con la media global (si los hubiera por edges).
3. Se aplica `cwt_morlet(var_s, scales)` usando la wavelet de Morlet compleja:
   `ψ(t) = exp(iω₀t) × exp(-t²/2) / √s`  con `ω₀ = 6`.
4. El scalograma es `SG[scale, t] = |W[scale, t]|²`.

**Qué ver:** en t < 400 el scalograma es oscuro (baja potencia, señal estacionaria). En t ≥ 400, a escalas medias y altas, aparece una franja brillante (naranja/amarillo en inferno) que indica que la varianza rodante está cambiando a esas frecuencias. La transición en t = 400 es claramente visible como un "frente de onda" vertical.

**Por qué escalas geométricas:** la wavelet de Morlet es un analizador tiempo-frecuencia logarítmico. Escalas pequeñas capturan cambios rápidos (alta frecuencia), escalas grandes capturan cambios lentos. Usar geomspace garantiza densidad uniforme en escala logarítmica.

---

#### Panel ④ — Surrogate calibration → signed log-ratio E⁺(t) − E⁻(t)

**Datos mostrados:**
- Área roja (`#E53935`, alpha 0.65): `E⁺(t)` — energía positiva (incrementos de potencia).
- Área azul (`#1565C0`, alpha 0.65): `−E⁻(t)` — energía negativa (decrementos de potencia), graficada hacia abajo.
- Línea horizontal gris en y = 0.

**Cómo se calcula:**

1. **Umbral surrogate** (simplificado para el plot):
   `thresh[scale] = percentile_95(SG[scale, 0:cp])` — el percentil 95 de la potencia en el baseline por cada escala. En el sistema real se usan 100 surrogados de Fourier (fase aleatoria) para estimar este umbral.

2. **Log-ratio con signo:**
   `Z[scale, t] = log((SG[scale,t] + ε) / (thresh[scale] + ε))`
   - Z > 0: la potencia supera el umbral → hay actividad anómala.
   - Z < 0: la potencia cae por debajo del umbral → hay supresión.

3. **Agregación entre escalas:**
   `E⁺(t) = mean_scale( max(Z, 0) )`
   `E⁻(t) = mean_scale( max(-Z, 0) )`
   `E_signed(t) = E⁺(t) - E⁻(t)`

**Qué ver:** en t < 400, E⁺ ≈ 0 y E⁻ ≈ 0 (el sistema está dentro de los límites de calibración). En t ≥ 400, E⁺ crece marcadamente (la varianza del proceso excede lo esperado bajo el régimen baseline). El área roja domina tras el punto de cambio.

**Por qué log-ratio:** un ratio simple `SG/thresh` tendría escala dependiente de los valores absolutos de la señal. El logaritmo lo vuelve simétrico: +1 = señal 2.7× encima del umbral, −1 = señal 2.7× debajo. Esto facilita la comparación de onset vs offset.

---

#### Panel ⑤ — dE/dt → peak detection (onset ▲ / offset ▽)

**Datos mostrados:**
- Traza gris-azul (`#546E7A`): derivada temporal de `E_signed`, suavizada.
- Triángulos rojos ▲ (onset): picos positivos de dE/dt.
- Triángulos azules ▽ (offset): picos negativos de dE/dt.
- Línea horizontal en y = 0.

**Cómo se calcula:**
1. `E_sm = uniform_filter1d(E_signed, size=15)` — suavizado para reducir ruido de alta frecuencia.
2. `dE = uniform_filter1d(gradient(E_sm), size=9)` — derivada numérica + segundo suavizado.
3. `find_peaks(dE, height=1.5×std(dE_baseline), distance=50)` — picos positivos (onset).
4. `find_peaks(-dE, height=1.5×std(dE_baseline), distance=50)` — picos negativos (offset).

**Por qué derivada en lugar de umbral directo:** un umbral directo sobre E(t) detecta el **estado** de alta energía, no la **transición**. Si el sistema cambia de régimen y permanece en él indefinidamente, E(t) se mantiene alta y generaría falsas alarmas continuas (cascada de FPs). La derivada captura el **momento exacto del cambio**, no la persistencia.

**Qué ver:** el triángulo rojo ▲ aparece en el instante en que E⁺ empieza a crecer rápidamente, ligeramente después de t = 400 (hay un lag pequeño por el suavizado y la ventana rodante). Este es el candidato de detección que pasa al gate siguiente.

---

#### Panel ⑥ — K-of-channels gate — consensus

**Datos mostrados:**
- 4 trazas de energía E_ch(t) por canal, desplazadas verticalmente (`offset = max(E_ch) × 0.6 × ch`) para mayor claridad.
  - Ch 1: rojo (`#E53935`)
  - Ch 2: naranja (`#F57C00`)
  - Ch 3: verde (`#2E7D32`)
  - Ch 4: azul (`#2E86AB`)
- Banda naranja semi-transparente: ventana temporal alrededor del pico candidato detectado en el panel ⑤.
- Anotación en caja blanca: `"K = 4 channels agree → detection KEPT"`.

**Cómo se calcula E_ch:**
Para cada canal `ch` independientemente:
1. Calcular varianza rodante de `X[:, ch]`.
2. Aplicar CWT Morlet con las mismas escalas.
3. Calcular log-ratio con umbral calibrado sobre el baseline del canal.
4. `E_ch[:, ch] = mean_scale( max(Z_ch, 0) )`.

**Lógica del gate:**
Un pico candidato en tiempo `τ` pasa el gate sólo si un número mínimo `k_channels_min` de canales muestran energía significativa en esa ventana temporal. En el sistema real se calcula:
- `Δ_ch = median(post) - median(pre)` — escalón por canal
- `z_ch = Δ_ch / σ_ch` — normalizado por MAD del pre-régimen
- Gate: `sum(z_ch > δ) ≥ k_channels_min`

En el plot se muestra la versión visual: las 4 curvas de energía por canal todas suben en t ≈ 400 simultáneamente, lo que valida el pico como multichannel consensus. La anotación `K = 4 channels agree → detection KEPT` resume la decisión.

**Por qué el gate importa:** un cambio en un único canal puede ser ruido, un artefacto residual, o un cambio local no estructural. Si los 4 (o K de N) canales lo muestran, es casi seguro un cambio sistémico en la dinámica causal subyacente.

---

### Cómo leer el plot en la presentación

> *"Cada fila es una etapa del pipeline. La señal entra arriba con un spike de artefacto. El MAD lo elimina. Los momentos rodantes transforman la señal en 4 estadísticas. El CWT convierte eso en un mapa tiempo-escala. El log-ratio nos dice cuándo esa potencia excede lo esperado por calibración. La derivada localiza el instante exacto del cambio. Y el gate final confirma que todos los canales están de acuerdo antes de declarar una detección."*

---

## Paleta de colores (referencia rápida)

| Elemento | Hex | Uso |
|---|---|---|
| `#2E86AB` | azul | señal principal, ch 4 |
| `#D32F2F` | rojo oscuro | líneas de cambio verdadero |
| `#FF5252` | rojo brillante | artefacto marcado |
| `#546E7A` | gris-azul | media, dE/dt |
| `#F57C00` | naranja | varianza, ch 2 |
| `#2E7D32` | verde | skewness, ch 3 |
| `#6A1B9A` | morado | kurtosis |
| `#E53935` | rojo | E⁺ onset, ch 1 |
| `#1565C0` | azul oscuro | E⁻ offset |
| `#E65100` | naranja oscuro | K-of-channels gate |
