# Guía del proyecto para el equipo

Bajo peso al nacer con el CNV del MINSA. Esta guía explica **qué tenemos, por qué está hecho así y qué falta**, para que todos podamos explicar el proyecto y responder preguntas en la exposición. No repite el problema ni el objetivo, que ya conocemos.

Todo está en el repositorio [github.com/jasnai10/AI_Project](https://github.com/jasnai10/AI_Project). Los números de esta guía salen de los notebooks ejecutados (01 a 05 en Colab y 06 en local).

---

## 1. Mapa del repositorio

```
AI_Project/
├── README.md                      # descripción, metodología, resultados, cómo ejecutar
├── ENTREGABLE_PARCIAL.md          # checklist de la semana 8
├── ENTREGABLE_FINAL.md            # checklist de la semana 15
└── proyecto/
    ├── data/raw/                  # dataset.zip original + descripción del dataset
    ├── data/processed/            # cnv_limpio.csv.gz (dataset limpio)
    ├── notebooks/
    │   ├── 01_eda.ipynb           # análisis exploratorio
    │   ├── 02_baseline.ipynb      # regresión logística simple
    │   ├── 03_rl_balanceada.ipynb # regresión logística con class_weight
    │   ├── 04_rl_smote.ipynb      # regresión logística con SMOTE-NC
    │   ├── 05_random_forest.ipynb # random forest balanceado
    │   └── 06_xgboost.ipynb       # XGBoost balanceado
    ├── src/preprocessing.py       # carga, limpieza, features, partición, preprocesador
    ├── results/
    │   ├── model_comparison.txt   # resumen de resultados y análisis detallado
    │   ├── metrics.csv            # las 6 métricas en test de cada modelo
    │   └── plots/                 # gráficos de cada modelo
    ├── papers/                    # un paper por integrante
    └── report.pdf                 # reporte académico (5 páginas)
```

**Sobre la estructura:** la estructura que proponen las indicaciones del curso (`02_modelos.ipynb`, `src/models.py`, `src/evaluation.py`, `requirements.txt`) es **opcional**, según nos indicó la profesora. La nuestra es así, y está explicada en `ENTREGABLE_FINAL.md`:

- **Un notebook por modelo** en lugar de uno solo: cada modelo tarda entre minutos y una hora en Colab; separados, se ejecutan y revisan por partes y uno que falla no arrastra a los demás.
- **Solo `src/preprocessing.py`:** es lo único que comparten todos los modelos (mismos datos, misma partición, mismo preprocesador). La implementación y la evaluación de cada modelo quedan en su notebook, donde se ven junto con sus resultados.
- **Sin `requirements.txt`:** los notebooks están preparados para Google Colab, que ya trae las librerías; la primera celda instala `imbalanced-learn` o `xgboost` si faltan.
- `results/` sí sigue la indicación: `metrics.csv`, `plots/` y `model_comparison.txt`.

---

## 2. Base del proyecto: dataset, papers y propuesta de modelos

1. **Dataset:** Certificados de Nacido Vivo (CNV) del MINSA, Plataforma Nacional de Datos Abiertos. 4 874 510 nacimientos, 22 variables, de 2015 a noviembre de 2025. Es público. La descripción está en `data/raw/Información del dataset.md`.
2. **Papers:** uno por integrante, sobre el mismo problema con otro dataset. Cada uno justifica uno o más de nuestros modelos:
   - Jair: Almeida et al. (2026), Brasil. Es la referencia principal: regresión logística y random forest, con y sin balanceo.
   - Luis: Ranjbar et al. (2023), Irán. Compara ocho modelos y XGBoost resulta el mejor.
   - Nilton: Liu et al. (2022), China. Estima el peso al nacer con varios modelos, entre ellos random forest y gradient boosting.
   - Pedro: Khan et al. (2022), Emiratos Árabes. Compara 30 algoritmos; la regresión logística con SMOTE es la mejor para clasificar.
3. **Propuesta de modelos:** regresión logística (balanceada y con SMOTE), random forest y XGBoost, que son los mejores modelos de esa literatura. La regresión logística simple es el baseline.

## 3. EDA y preprocesamiento: qué se decidió y por qué

El EDA evalúa las 22 columnas con los mismos **seis criterios** para decidir qué variables entran al modelo:

| # | Criterio | Pregunta | Cómo se mide |
|---|---|---|---|
| 1 | Validez | ¿Existe cuando se usaría el modelo y no es consecuencia del parto ni una medida del recién nacido? | Diccionario de datos y cruce con el target |
| 2 | Calidad | ¿El dato es creíble y consistente? | Códigos de faltante, rangos, estabilidad por año, si el faltante es aleatorio |
| 3 | Redundancia | ¿Otra variable incluida ya lo dice? | Correlación de Spearman y V de Cramér |
| 4 | Asociación propia | ¿Sigue asociada al bajo peso después de controlar por la duración del embarazo? | Razón observada/esperada ajustada, con intervalo de confianza del 95 % |
| 5 | Costo | ¿Cuántas filas o categorías cuesta? | Casos completos y número de categorías |
| 6 | Uso | ¿Es apropiada como entrada y qué decisión se tomaría con ella? | Decisión del equipo, declarada por escrito |

**Regla de decisión:**

- Una variable **entra** si cumple validez, calidad y uso (1, 2 y 6, que son requisitos), tiene asociación propia (4) y no es redundante (3). El costo (5) desempata.
- Una variable **sale** si falla un requisito, o si no tiene asociación propia y además cuesta filas o es redundante.
- Hay asociación propia si en al menos una comparación la razón ajustada es ≥ 1.10 o ≤ 0.91 y su intervalo de confianza excluye el 1.

Lo importante para defender:

**Problemas de calidad encontrados**

- **388 filas rotas:** algunas ocupaciones contienen `;`, que es el separador del CSV, y desplazan las columnas. Se reparan al cargar, sin perder filas.
- **Los faltantes vienen codificados** (`IGNORADO`, `-1`, `99`), no como celdas vacías, así que `isna()` no los detecta. Se convierten a `NaN`.
- **Valores fuera de rango:** por ejemplo, 558 pesos de exactamente 300 g (probablemente 3000 g mal digitados). Se convierten a `NaN`; los registros sin peso válido se eliminan (quedan 4 871 486).
- **974 duplicados:** se mantienen, porque sin un ID no se puede distinguir un error de dos nacimientos reales (8 % son partos dobles).

**Variables excluidas**

| Variable | Motivo |
|---|---|
| `PESO_NACIDO` | Es el target (bajo peso = peso < 2500 g) |
| `TALLA_NACIDO` | Medida del mismo recién nacido (correlación 0.72 con el peso): sería predecir el peso con el peso |
| `Atiende_Parto`, `Condicion_Parto`, `Lugar_Nacido`, `Ipress` | Describen el parto, en parte como consecuencia del bajo peso (un prematuro tiene más probabilidad de cesárea) |
| `nacmuer_abort_madre` | 8.6 % de faltantes que varían por departamento (5 % a 21 %) y por año: no es aleatorio. Incluirla bajaría los casos completos de 97.8 % a 89.6 % |
| `Estado_Civil` | Registro poco confiable: el 79 % de las madres de 17 años o menos figura como `IGNORADO` |
| `Hijos_fallec_madre` | 97 % con `-1` y significado ambiguo (no existe la opción 0) |
| `Hijos_vivo_madre` | Redundante con `Num_embar_madre` (correlación 0.88) |
| `DESC_OCUPACION` | Texto libre |

**Las 9 features del modelo:** `DUR_EMB_PARTO` (semanas de gestación), `Tipo_Parto` (único o múltiple), `departamento`, `edad_grupo` (edad de la madre agrupada), `Num_embar_madre`, `Nivel_Instruccion`, `Financiador_Parto`, `sexo_nacido` y `pais_grupo` (Perú, Venezuela u otro).

- Las 8 distintas de la duración **conservan un efecto propio** al controlar por ella (estandarización indirecta por semanas de gestación). Por ejemplo, un parto múltiple tiene 1.55 veces el riesgo esperado a igual duración del embarazo.
- `pais_grupo` se interpreta como un marcador de contexto, no como un efecto de la nacionalidad.
- `Tipo_Parto` es único o múltiple, **no** cesárea o vaginal.

**Datos finales:** 4 764 960 nacimientos con las 9 variables completas (97.81 % de los que tienen peso válido). Prevalencia de bajo peso: 6.38 %, una proporción de 1 a 15.

**Transformación para los modelos:** one-hot para las 7 categóricas y estandarización para las 2 numéricas, lo que da 53 columnas. El preprocesador **se ajusta solo con train**, para no filtrar información de validación ni de test.

**Partición** (`dividir_temporal` en `src/preprocessing.py`):

| Conjunto | Años | Registros | Prevalencia | Para qué se usa |
|---|---|---|---|---|
| Train | 2015–2023 (80 %) | 3 237 829 | 6.32 % | Entrenar |
| Validación | 2015–2023 (20 %) | 809 458 | 6.32 % | Elegir hiperparámetros |
| Test | 2024–2025 | 717 673 | 6.81 % | Evaluación final, una sola vez |

Train y validación se separan al azar y estratificados por bajo peso. El test no incluye diciembre de 2025 (el corte del dataset es el 30/11/2025).

---

## 4. Los modelos

| Notebook | Modelo | De qué paper sale | Cómo trata el desbalance |
|---|---|---|---|
| 02 | Regresión logística simple (**baseline**) | Almeida (modelo sin balancear) | No lo trata |
| 03 | Regresión logística balanceada | Almeida (su mejor modelo) | `class_weight="balanced"`: cada caso de bajo peso pesa unas 15 veces más al entrenar |
| 04 | Regresión logística con SMOTE-NC | Khan (su mejor clasificador) | Crea casos sintéticos de bajo peso en train |
| 05 | Random forest balanceado | Almeida, Ranjbar y Khan | `class_weight="balanced"` |
| 06 | XGBoost balanceado | Ranjbar (su mejor modelo) | `scale_pos_weight` ≈ 14.8, el equivalente de `class_weight="balanced"` |

**Por qué un baseline:** es el punto de comparación. Un modelo solo vale la pena si mejora lo que da una regresión logística sin ningún tratamiento.

**Por qué SMOTE-NC y no SMOTE:** SMOTE interpola entre vecinos, lo que no tiene sentido en variables categóricas (daría "departamento = 0.37"). SMOTE-NC interpola las numéricas y, en las categóricas, toma la categoría más frecuente entre los vecinos. El sobremuestreo se aplica **solo al train**; validación y test conservan la distribución real.

Todos los notebooks tienen las mismas secciones: 1 Configuración, 2 Datos y partición temporal, 3 Ajuste de hiperparámetros (en el baseline, el modelo), 4 Evaluación en test, 5 Interpretación y 6 Limitaciones.

---

## 5. Protocolo común: decisiones y por qué

### 5.1 Ajuste de hiperparámetros

- Se prueba una **grilla** de combinaciones; cada una se entrena con train y se mide en **validación**. Se elige la de mayor **AUC-PR**.
- **Empate:** si dos combinaciones quedan a menos de 0.001, se elige la más simple (más regularización, menor proporción de SMOTE, etc.). Diferencias tan pequeñas son ruido.
- **Solo hiperparámetros del modelo**, no del algoritmo. Un hiperparámetro del modelo cambia lo que se aprende (`C`, tipo de penalización, profundidad de los árboles); uno del algoritmo solo cambia cómo se calcula (el solver, el paralelismo). Por eso en las regresiones el solver quedó fijo en `saga`, el único que sirve para L1 y L2 sin penalizar el intercepto.
- **El test no se usa para ajustar nada.**
- Se grafica AUC-PR en train y en validación para cada combinación: si train es mucho mayor que validación, hay sobreajuste.

### 5.2 Por qué AUC-PR

- Con 6.8 % de bajo peso, la **accuracy engaña**: predecir siempre "peso normal" ya da 93 %.
- El **AUC-ROC** mide qué tan bien el modelo ordena: la probabilidad de que un caso de bajo peso reciba más riesgo que uno normal. 0.5 es azar y 1 es perfecto.
- El **AUC-PR** resume la precisión promedio mientras se van encontrando los casos. Su valor de azar es la prevalencia (0.068), no 0.5, así que es más exigente cuando la clase de interés es rara. Por eso es la métrica principal.
- Ninguna de las dos depende del umbral.

### 5.3 Umbral de 0.5

- El modelo da una probabilidad; el umbral decide a partir de cuánto se marca como bajo peso.
- Se usa 0.5, sin calibrar, **como en Almeida, Khan y Ranjbar**. Así las métricas son comparables con las de los papers.
- Precisión, recall, F1 y accuracy dependen de ese umbral; AUC-ROC y AUC-PR no.

### 5.4 Métricas reportadas (en test)

- **Recall:** de los bebés con bajo peso, qué proporción detecta el modelo.
- **Precisión:** de las alertas del modelo, qué proporción son casos reales.
- **F1:** equilibrio entre precisión y recall.
- **Accuracy, AUC-ROC y AUC-PR:** ver arriba.

### 5.5 Cómo se elige el mejor modelo

Cada métrica favorece a un modelo distinto, así que no se elige por mayoría. Hay un orden fijado de antemano, basado en Almeida y Ranjbar:

1. **Discriminación** (AUC-PR y AUC-ROC).
2. **Recall**, entre modelos con discriminación parecida, porque en salud pública lo prioritario es no perder casos.
3. **F1 y precisión**, para describir el costo en falsas alarmas.

La comparación final se hace en el test, como en los papers. Lo declaramos como limitación.

---

## 6. Resultados (test 2024–2025)

| Modelo | Accuracy | Recall | F1 | Precisión | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Baseline | 0.957 | 0.473 | 0.600 | 0.820 | 0.908 | 0.677 |
| RL balanceada | 0.865 | 0.809 | 0.449 | 0.310 | 0.908 | 0.677 |
| RL SMOTE-NC | 0.948 | 0.630 | 0.624 | 0.618 | 0.907 | 0.677 |
| Random forest balanceado | 0.880 | 0.793 | 0.475 | 0.339 | 0.911 | 0.679 |
| **XGBoost balanceado** | 0.873 | 0.803 | 0.462 | 0.325 | **0.913** | **0.682** |

Hiperparámetros elegidos:

| Modelo | Grilla | Elegido | Tiempo |
|---|---|---|---|
| Baseline | Sin ajuste (`C=1`, L2) | — | ≈ 13 s |
| RL balanceada | `C` {1e-5 … 1} × L1/L2 (12) | L2, `C=0.0001` | ≈ 12 min |
| RL SMOTE-NC | proporción {0.25, 0.5} × `C` {1e-4, 1e-3} × L1/L2 (8) | 0.25, L2, `C=0.0001` | ≈ 1 h |
| Random forest | `min_samples_leaf` {10, 50, 200} × `max_features` {"sqrt", 0.5} × `max_samples` {0.1, 0.3} (12), 50 árboles | 50, 0.5, 0.3 | ≈ 35 min |
| XGBoost | `max_depth` {4, 6, 8} × `min_child_weight` {1, 100} × `scale_pos_weight` {1, 14.8} (12); árboles con *early stopping* (hasta 2000) | 4, 100, 14.8; 917 árboles | ≈ 64 min (en local) |

Los tiempos de 02 a 05 son en Colab; el 06 se ejecutó en local, así que su tiempo no es directamente comparable.

### 6.1 Cómo explicar los resultados

1. **Mejor modelo: XGBoost balanceado.** Tiene el AUC-ROC (0.913) y el AUC-PR (0.682) más altos de los cinco, aunque por poco: 0.005 y 0.004 más que el baseline, y 0.002 y 0.003 más que el random forest, que queda segundo. La ventaja también aparece en validación. Detecta el 80 % de los bebés con bajo peso (39 271 de 48 907) con unas 2 falsas alarmas por caso detectado.
2. **Todos discriminan parecido.** El AUC-PR va de 0.677 a 0.682, unas 10 veces el azar. Con estas 9 variables, ni el balanceo ni un modelo no lineal cambian mucho la capacidad de separar las clases.
3. **El balanceo cambia el punto de operación, no la discriminación.** Las tres regresiones tienen el mismo AUC; lo que cambia es cuántos casos marcan con el umbral 0.5:
   - El baseline marca pocos: precisión alta (0.82) pero pierde más de la mitad de los casos (recall 0.47).
   - La balanceada marca muchos: detecta el 81 % pero con unas 2 falsas alarmas por caso detectado.
   - SMOTE-NC queda en el medio y tiene el mejor F1.

   Es el mismo patrón que reporta Almeida.
4. **Por qué los modelos de árboles son un poco mejores:** XGBoost y el random forest son los dos modelos no lineales. Captan interacciones (por ejemplo, que el efecto de las semanas de gestación cambie según el tipo de parto o la región) que la regresión logística, al sumar efectos independientes, no representa.
   - **XGBoost** entrena los árboles uno tras otro y cada uno corrige los errores de los anteriores (*boosting*). Cada corte puede usar las 53 columnas y el número de árboles se ajusta con *early stopping*. Con un recall casi igual al de la RL balanceada, da unas 6 200 falsas alarmas menos.
   - **El random forest** promedia 50 árboles independientes y cada corte ve solo una parte de las variables; por eso queda levemente por debajo.
   - A cambio, ambos tardan mucho más en entrenar y son menos interpretables.
5. **SMOTE-NC no aporta.** Khan lo usó con 89 casos de bajo peso; nosotros tenemos más de 200 000 casos reales en train, así que los sintéticos no añaden información. Además fue el modelo más lento (≈ 1 h), cuando `class_weight` logra lo mismo en minutos.
6. **La precisión baja de XGBoost (0.32) y del random forest (0.34) no es un error.** Es el efecto de balancear con el umbral 0.5: el modelo prioriza detectar casos.
7. **Calibración:** solo el baseline da probabilidades realistas. Los balanceados sobreestiman el riesgo, porque se entrenaron como si la clase fuera más frecuente. Sirven para ordenar y alertar, no como probabilidad absoluta.
8. **Estabilidad en el tiempo:** el desempeño en 2024–2025 es igual o mejor que en validación (AUC-ROC del baseline 0.904 → 0.908; de XGBoost 0.909 → 0.913), así que los modelos generalizan a años futuros.

### 6.2 Limitaciones (decirlas antes de que las pregunten)

- **`DUR_EMB_PARTO` se conoce al momento del parto.** El modelo sirve para anticipar, antes del pesaje, qué recién nacidos pueden necesitar atención; no es una predicción prenatal.
- Umbral fijo en 0.5: precisión, recall y F1 dependen de él.
- El modelo final se elige comparando en el test, como los papers; puede dar una estimación algo optimista.
- El test tiene más prevalencia (6.81 %) que train (6.32 %).
- El CNV no registra variables clínicas (controles prenatales, anemia, hipertensión, IMC).
- Ajuste acotado por cómputo: una sola partición de validación, una semilla, 50 árboles en el random forest, `learning_rate` fijo en XGBoost y SMOTE-NC sin probar el balance total (se quedaba sin memoria).
- No se estimaron intervalos de confianza: con diferencias de milésimas, no se puede asegurar que el orden de los modelos se mantenga en otros años.

---

## 7. Preguntas probables

- **¿Por qué no usan accuracy?** Con 6.8 % de prevalencia, predecir siempre "normal" da 93 %. El baseline tiene la mejor accuracy (0.957) y es el que más casos pierde.
- **¿Por qué partición temporal y no aleatoria?** Porque el modelo se usaría con nacimientos futuros. Entrenar con el pasado y evaluar con los años siguientes es más realista y más exigente.
- **¿El test se usó para elegir hiperparámetros?** No. Los hiperparámetros se eligieron en validación; el test se usó una sola vez por modelo.
- **¿Hay fuga de información?** No. El preprocesador y SMOTE-NC se ajustan solo con train, y se excluyeron las variables que son consecuencia del parto o del peso (talla, tipo de atención, condición del parto).
- **¿Por qué el umbral es 0.5?** Es el que usan los papers de referencia, así las métricas son comparables.
- **¿Por qué la diferencia entre modelos es tan pequeña?** Porque el límite lo pone la información de las 9 variables, no el modelo. Para mejorar claramente harían falta variables clínicas que el CNV no tiene.
- **¿Hay sobreajuste?** En las regresiones no: train y validación dan casi lo mismo. En el random forest, las hojas de 10 registros sí sobreajustaban, por eso se eligieron hojas de 50. En XGBoost fue leve (brecha de AUC-PR ≤ 0.007), controlado por el *early stopping*.
- **¿Por qué no probaron SMOTE al 100 % como Khan?** Generar 2.4 millones de casos sintéticos excedía la memoria de Colab. Con 0.25 y 0.5 ya se ve que no mejora la discriminación.
- **¿Por qué solo 50 árboles en el random forest?** Por tiempo: con 3.2 millones de registros, cada bosque tarda entre 1 y 6.5 minutos en Colab.
- **¿Por qué en XGBoost las 12 combinaciones empataron?** Porque el *early stopping* compensa la profundidad con el número de árboles: los árboles poco profundos necesitan más árboles (hasta 917) y los profundos se detienen antes (117 a 227). Todas llegan al mismo techo; el desempate eligió la más simple y, entre la versión con y sin balanceo, la de mayor recall.
- **¿El balanceo afecta a XGBoost?** No cambia su discriminación (diferencias de 0.0002 en AUC-PR); solo mueve el punto de operación, igual que en la regresión logística.
- **¿Cómo se comparan con los papers?** Almeida obtuvo AUC-ROC 0.85 con la RL balanceada y Ranjbar 0.79 con XGBoost; nosotros 0.913 con XGBoost. Como en Ranjbar, XGBoost queda primero. No son directamente comparables (otros países, variables y prevalencias), pero estamos en el rango de la literatura.
- **¿Qué modelo recomendarían?** XGBoost balanceado, por discriminación y recall. Si se quisiera menos falsas alarmas, SMOTE-NC o el baseline: elegir el modelo también es elegir cuántas alertas se pueden atender.

---

## 8. Lo que falta

| Pendiente | Responsable | Notas |
|---|---|---|
| Presentación (máximo 5 minutos, resultados y conclusiones) | Nilton Condori | Usar la sección 6.1 como guion |
| Cerrar el checklist de `ENTREGABLE_FINAL.md` | Por definir | Cuando la presentación esté lista |

---

## 9. Contribución de cada integrante

Es la misma tabla que va en el reporte (`report.pdf`):

| Integrante | Entregable parcial | Entregable final |
|---|---|---|
| Aguilera Inca, Jair Sneider | Baseline, paper de Almeida y archivos Markdown solicitados | Regresión logística balanceada y random forest balanceado |
| Araujo Vigo, Luis | Paper de Ranjbar y descripción del dataset | XGBoost y reporte académico |
| Condori Quispe, Nilton | Preprocesamiento (`src/preprocessing.py`) y paper de Liu | Presentación y archivos Markdown del repositorio |
| Vilcayauri Campos, Pedro | Análisis exploratorio (EDA) y paper de Khan | Regresión logística con SMOTE-NC |
