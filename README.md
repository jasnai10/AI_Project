# Predicción de bajo peso al nacer en el Perú con datos del CNV (MINSA)

**Inteligencia Artificial Aplicada (1INF62) — Pontificia Universidad Católica del Perú, 2026-2**

**Integrantes:** Aguilera, Jair · Araujo, Luis · Condori, Nilton · Vilcayauri, Pedro

## Problema

El bajo peso al nacer (< 2500 g, criterio de la OMS) está asociado a mayor morbilidad y mortalidad neonatal. En el registro de Certificados de Nacido Vivo (CNV) del MINSA, entre 2015 y 2025, afecta al 6.38% de los nacimientos. Identificar con anticipación los embarazos con mayor riesgo permitiría priorizar el seguimiento prenatal, pero los registros administrativos rara vez se han usado con ese fin en el Perú.

## Objetivo

Evaluar modelos de aprendizaje automático para **clasificar** a los recién nacidos según si tienen bajo peso al nacer (variable objetivo binaria `bajo_peso`: 1 si el peso es menor de 2500 g, 0 en otro caso) a partir de los registros del CNV, con un único conjunto de **9 variables**: las características de la madre y del parto registradas en el certificado, más la duración del embarazo (`DUR_EMB_PARTO`).

La duración del embarazo es, por mucho, la variable dominante (AUC univariado de 0.89, frente a 0.58 o menos de cualquier otra), y se conoce **al momento del parto, no antes**. Por eso el modelo predice a partir de lo registrado al nacer y antes del pesaje, no con anticipación al parto; esta dependencia se declara en el EDA y la asociación de las demás variables se mide condicionada a ella.

## Dataset

Certificados de Nacido Vivo del MINSA (Plataforma Nacional de Datos Abiertos), corte al 30/11/2025: 4 874 510 registros y 22 variables, de 2015 a 2025. La descripción completa del origen, las variables y los permisos está en `data/raw/`.

## Metodología

1. **Análisis exploratorio** (`notebooks/01_eda.ipynb`): calidad de datos, distribución del target, relación de cada variable con el bajo peso y justificación de las decisiones de preprocesamiento. Las 22 columnas se evalúan con seis criterios: validez, calidad, redundancia, asociación propia (condicionada a la duración del embarazo), costo y uso.
2. **Preprocesamiento** (`src/preprocessing.py`): implementa las decisiones del EDA y es compartido por todos los notebooks de modelado. Las principales son:
   - Reparación de 388 filas con el separador `;` dentro del campo de ocupación.
   - Conversión de los códigos de faltante (`IGNORADO`, `-1`, `99`) y de los valores fuera de rango a valores ausentes, con eliminación de casos completos sobre las 9 variables (se conserva el 97.81 % de los registros con peso válido).
   - **Features del modelo (9):** `DUR_EMB_PARTO`, `Tipo_Parto`, `departamento`, `Edad_Madre` (como grupo de edad), `Num_embar_madre`, `Nivel_Instruccion`, `Financiador_Parto`, `sexo_nacido` y `pais_grupo`. `pais_grupo` se mantiene por decisión del equipo y se interpreta como un marcador de contexto, no como un efecto de la nacionalidad.
   - Exclusión de las variables que describen el parto o el desenlace (`TALLA_NACIDO`, `Atiende_Parto`, `Condicion_Parto`, `Lugar_Nacido`, `Ipress`), de `nacmuer_abort_madre` (faltante no aleatorio que cuesta el 79 % de las filas perdidas), de `Estado_Civil` (baja validez del registro), de `Hijos_fallec_madre` (código `-1` ambiguo), de `Hijos_vivo_madre` (redundante con el número de embarazos) y de `DESC_OCUPACION` (texto libre).
   - Agrupación de categorías poco frecuentes y construcción de `departamento` y `pais_grupo`.
   - One-hot para las 7 variables categóricas y estandarización para las 2 numéricas (53 columnas), ajustado solo con el entrenamiento.
   - **Partición temporal:** entrenamiento y validación con 2015–2023 (80 % / 20 % al azar, estratificado) y test con 2024–2025, que se usa una sola vez al final. El corte del CNV es el 30/11/2025, así que 2025 no incluye diciembre.
3. **Datos procesados** (`data/processed/cnv_limpio.csv.gz`): resultado de la limpieza, con 4 871 486 registros (los que tienen peso válido).
4. **Modelos** (`notebooks/02_baseline.ipynb` a `notebooks/06_xgboost.ipynb`): un notebook por modelo, todos con el mismo protocolo:
   - Hiperparámetros del modelo ajustados con una grilla evaluada por **AUC-PR en validación**.
   - **Umbral de decisión fijo en 0.5**, sin calibrar, como en los papers de referencia.
   - Evaluación única en el test 2024–2025 con accuracy, precisión, recall, F1-score, AUC-ROC y AUC-PR.
5. **Comparación** (`results/`): tabla de métricas, gráficos y análisis de qué modelo funcionó mejor, por qué, limitaciones e insights.

## Modelos

### Mejores modelos reportados en la literatura revisada

| Paper | Tarea | Mejor modelo reportado |
|---|---|---|
| Almeida et al. (2026) | Clasificación de bajo peso | Regresión logística balanceada, frente a random forest |
| Khan et al. (2022) | Clasificación de bajo peso | Regresión logística con sobremuestreo (SMOTE), entre 30 algoritmos |
| Khan et al. (2022) | Estimación del peso al nacer | Random forest, entre 30 algoritmos |
| Liu et al. (2022) | Estimación del peso al nacer | Ensamble de votación y SVR, por delante de la regresión lineal, random forest, AdaBoost y gradient boosting |
| Ranjbar et al. (2023) | Clasificación de bajo peso | XGBoost como mejor modelo global; la red feedforward, random forest y XGBoost obtuvieron los mayores AUROC entre ocho modelos |

### Modelos implementados

| Notebook | Modelo | Justificación |
|---|---|---|
| `02_baseline.ipynb` | Regresión logística simple (baseline) | Punto de referencia sin tratamiento del desbalance; modelo sin balancear de Almeida et al. (2026). |
| `03_rl_balanceada.ipynb` | Regresión logística balanceada (`class_weight="balanced"`) | Mejor clasificador en Almeida et al. (2026). |
| `04_rl_smote.ipynb` | Regresión logística con sobremuestreo SMOTE-NC | Mejor clasificador en Khan et al. (2022); se usa SMOTE-NC porque casi todas las variables son categóricas. |
| `05_random_forest.ipynb` | Random forest balanceado | Único modelo presente en los cuatro papers: mejor modelo para estimar el peso en Khan et al. (2022), entre los tres con mayor AUROC en Ranjbar et al. (2023) y modelo de comparación en Almeida et al. (2026). |
| `06_xgboost.ipynb` | XGBoost balanceado (`scale_pos_weight`) | Mejor modelo para predecir el bajo peso según Ranjbar et al. (2023). El balanceo se eligió en la grilla de hiperparámetros. |

## Resultados

Resultados en el test 2024–2025 (717 673 nacimientos, 6.81 % con bajo peso), con umbral 0.5. En negrita, el mejor valor de cada columna:

| Modelo | Accuracy | Recall | F1 | Precisión | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Baseline: regresión logística simple | **0.957** | 0.473 | 0.600 | **0.820** | 0.908 | 0.677 |
| Regresión logística balanceada | 0.865 | **0.809** | 0.449 | 0.310 | 0.908 | 0.677 |
| Regresión logística con SMOTE-NC | 0.948 | 0.630 | **0.624** | 0.618 | 0.907 | 0.677 |
| Random forest balanceado | 0.880 | 0.793 | 0.475 | 0.339 | 0.911 | 0.679 |
| XGBoost balanceado | 0.873 | 0.803 | 0.462 | 0.325 | **0.913** | **0.682** |

**XGBoost balanceado es el modelo con mejor capacidad de discriminación** (AUC-ROC 0.913 y AUC-PR 0.682, los más altos de los cinco) y detecta el 80 % de los casos de bajo peso. Su ventaja es pequeña: 0.005 en AUC-ROC y 0.004 en AUC-PR sobre el baseline. Los cinco modelos discriminan de forma parecida, y lo que más cambia entre ellos es el equilibrio entre casos detectados y falsas alarmas. El resumen completo y el análisis detallado están en [`results/model_comparison.txt`](proyecto/results/model_comparison.txt), y el reporte académico, en [`report.pdf`](proyecto/report.pdf).

## Estructura del repositorio

La estructura propuesta en las indicaciones del curso es opcional; las adaptaciones se explican en `ENTREGABLE_FINAL.md`.

```
AI_Project/
├── README.md
├── ENTREGABLE_PARCIAL.md
├── ENTREGABLE_FINAL.md
└── proyecto/
    ├── data/
    │   ├── raw/
    │   │   ├── dataset.zip                  # dataset original del CNV
    │   │   └── Información del dataset.md   # origen, variables y permisos
    │   └── processed/
    │       └── cnv_limpio.csv.gz            # dataset limpio generado con src/preprocessing.py
    ├── notebooks/
    │   ├── 01_eda.ipynb                     # análisis exploratorio
    │   ├── 02_baseline.ipynb                # regresión logística simple
    │   ├── 03_rl_balanceada.ipynb           # regresión logística balanceada
    │   ├── 04_rl_smote.ipynb                # regresión logística con SMOTE-NC
    │   ├── 05_random_forest.ipynb           # random forest balanceado
    │   └── 06_xgboost.ipynb                 # XGBoost balanceado
    ├── src/
    │   └── preprocessing.py                 # carga, limpieza, features, partición y preprocesador
    ├── results/
    │   ├── model_comparison.txt             # resumen de resultados y análisis detallado
    │   ├── metrics.csv                      # tabla comparativa de métricas en test
    │   └── plots/                           # gráficos de cada modelo
    ├── papers/
    │   ├── paper_Jair_Aguilera.pdf
    │   ├── paper_luis_araujo.pdf
    │   ├── paper_Nilton_Condori.pdf
    │   └── paper_Pedro_Vilcayauri.pdf
    ├── report.pdf                           # reporte académico (paper de 5 páginas)
    └── presentation.pdf                     # diapositivas de la presentación
```

## Cómo ejecutar

Los notebooks están preparados para **Google Colab** y ya están guardados con sus resultados, así que se pueden revisar sin ejecutarlos. Se abren directamente desde estos enlaces (se requiere una cuenta de Google):

| Notebook | Contenido | Tiempo aproximado en Colab |
|---|---|---|
| [`01_eda.ipynb`](https://colab.research.google.com/github/jasnai10/AI_Project/blob/main/proyecto/notebooks/01_eda.ipynb) | Análisis exploratorio | Pocos minutos |
| [`02_baseline.ipynb`](https://colab.research.google.com/github/jasnai10/AI_Project/blob/main/proyecto/notebooks/02_baseline.ipynb) | Regresión logística simple | Pocos minutos |
| [`03_rl_balanceada.ipynb`](https://colab.research.google.com/github/jasnai10/AI_Project/blob/main/proyecto/notebooks/03_rl_balanceada.ipynb) | Regresión logística balanceada | ≈ 15 min |
| [`04_rl_smote.ipynb`](https://colab.research.google.com/github/jasnai10/AI_Project/blob/main/proyecto/notebooks/04_rl_smote.ipynb) | Regresión logística con SMOTE-NC | ≈ 1 h |
| [`05_random_forest.ipynb`](https://colab.research.google.com/github/jasnai10/AI_Project/blob/main/proyecto/notebooks/05_random_forest.ipynb) | Random forest balanceado | ≈ 40 min |
| [`06_xgboost.ipynb`](https://colab.research.google.com/github/jasnai10/AI_Project/blob/main/proyecto/notebooks/06_xgboost.ipynb) | XGBoost balanceado | Más de 1 h (la grilla tomó ≈ 64 min en local) |

Con el notebook abierto, ejecutar todas las celdas desde **Entorno de ejecución → Ejecutar todas**, con el entorno estándar de CPU. La primera celda descarga automáticamente el repositorio en la sesión de Colab e instala `imbalanced-learn` o `xgboost` si faltan, así que no es necesario subir ningún archivo. Cada notebook es independiente: carga el dataset original y aplica el mismo preprocesamiento, así que se pueden ejecutar en cualquier orden. En los notebooks largos conviene mantener la pestaña abierta hasta que terminen.

Si los enlaces no funcionan, también se pueden abrir manualmente:

1. Entrar a [Google Colab](https://colab.research.google.com/) y crear un cuaderno con **Nuevo cuaderno**.
2. En el menú superior del cuaderno, ir a **Archivo → Abrir cuaderno** y elegir **GitHub** en el panel izquierdo.
3. Pegar `https://github.com/jasnai10/AI_Project` en el campo de búsqueda, verificar que la rama sea `main` y hacer clic en el notebook en la lista **Ruta**.

## Referencias

- Almeida, A. V. A., Santos, A. O. R., & Nascimento, L. F. C. (2026). Predictive modeling with machine learning for low birth weight in the state of São Paulo: a retrospective population-based study. *São Paulo Medical Journal*. https://doi.org/10.1590/1516-3180.2025.3373.04052026
- Khan, W., Zaki, N., Masud, M. M., Ahmad, A., Ali, L., Ali, N., & Ahmed, L. A. (2022). Infant birth weight estimation and low birth weight classification in United Arab Emirates using machine learning algorithms. *Scientific Reports, 12*, 12110. https://doi.org/10.1038/s41598-022-14393-6
- Liu, Z., Han, N., Su, T., Ji, Y., Bao, H., Zhou, S., Luo, S., Wang, H., Liu, J., & Wang, H.-J. (2022). Interpretable machine learning to identify important predictors of birth weight: A prospective cohort study. *Frontiers in Pediatrics, 10*, 899954. https://doi.org/10.3389/fped.2022.899954
- Ranjbar, A., Montazeri, F., Farashah, M. V., Mehrnoush, V., Darsareh, F., & Roozbeh, N. (2023). Machine learning-based approach for predicting low birth weight. *BMC Pregnancy and Childbirth, 23*, 803. https://doi.org/10.1186/s12884-023-06128-w
