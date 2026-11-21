# Predicción de bajo peso al nacer en el Perú con datos del CNV (MINSA)

**Inteligencia Artificial Aplicada (1INF62) — Pontificia Universidad Católica del Perú, 2026-2**

**Integrantes:** Aguilera, Jair · Araujo, Luis · Condori, Nilton · Vilcayauri, Pedro

## Problema

El bajo peso al nacer (< 2500 g, criterio de la OMS) está asociado a mayor morbilidad y mortalidad neonatal. En el registro de Certificados de Nacido Vivo (CNV) del MINSA, entre 2015 y 2025, afecta al 6.38% de los nacimientos. Identificar con anticipación los embarazos con mayor riesgo permitiría priorizar el seguimiento prenatal, pero los registros administrativos rara vez se han usado con ese fin en el Perú.

## Objetivo

Evaluar modelos de aprendizaje automático para predecir el bajo peso al nacer a partir de los registros del CNV, con un único conjunto de **9 variables**: las características de la madre y del parto registradas en el certificado, más la duración del embarazo (`DUR_EMB_PARTO`).

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
   - **Partición temporal:** entrenamiento y validación con 2015–2023 (80 % / 20 % al azar, estratificado) y test con 2024–2025, que se usa una sola vez al final. El preprocesador se ajusta solo con el entrenamiento y el umbral de decisión se calibra en validación. El corte del CNV es el 30/11/2025, así que 2025 no incluye diciembre. El split aleatorio 80/20 queda como referencia secundaria.
3. **Datos procesados** (`data/processed/cnv_limpio.csv.gz`): resultado de la limpieza, con 4 871 486 registros (los que tienen peso válido).
4. **Baseline** (`notebooks/02_modelos.ipynb`): regresión logística con y sin balanceo de clases, siguiendo el diseño de Almeida et al. (2026), evaluada con precisión, recall, F1-score, AUC-ROC y AUC-PR sobre el test temporal. Como la prevalencia de bajo peso cambia entre train/validación (6.32 %) y test (6.81 %), las métricas dependientes de la prevalencia no son comparables con las de un split aleatorio.

## Propuesta de modelos

### Mejores modelos reportados en la literatura revisada

| Paper | Tarea | Mejor modelo reportado |
|---|---|---|
| Almeida et al. (2026) | Clasificación de bajo peso | Regresión logística balanceada, frente a random forest |
| Khan et al. (2022) | Clasificación de bajo peso | Regresión logística con sobremuestreo (SMOTE), entre 30 algoritmos |
| Khan et al. (2022) | Estimación del peso al nacer | Random forest, entre 30 algoritmos |
| Liu et al. (2022) | Estimación del peso al nacer | Ensamble de votación y SVR, por delante de la regresión lineal, random forest, AdaBoost y gradient boosting |
| Ranjbar et al. (2023) | Clasificación de bajo peso | XGBoost como mejor modelo global; la red feedforward, random forest y XGBoost obtuvieron los mayores AUROC entre ocho modelos |

### Modelos seleccionados

| Modelo | Justificación |
|---|---|
| **Regresión logística** | Mejor clasificador en Almeida et al. (2026) y en Khan et al. (2022). Es el baseline, ya implementado en `notebooks/02_modelos.ipynb`. |
| **Random forest** | Único modelo presente en los cuatro papers: mejor modelo para estimar el peso en Khan et al. (2022), entre los tres con mayor AUROC en Ranjbar et al. (2023) y modelo de comparación en Almeida et al. (2026). |
| **XGBoost** | Mejor modelo para predecir el bajo peso según Ranjbar et al. (2023), considerando accuracy, precisión, recall y F1-score. |

Los tres modelos se entrenarán con y sin balanceo de clases y se evaluarán sobre el mismo test temporal (2024–2025) y con las mismas métricas que el baseline.

## Estructura del repositorio

```
proyecto/
├── README.md
├── ENTREGABLE_PARCIAL.md
├── data/
│   ├── raw/
│   │   ├── dataset.zip                  # dataset original del CNV
│   │   └── Información del dataset.md   # origen, variables y permisos
│   └── processed/
│       └── cnv_limpio.csv.gz            # dataset limpio generado con src/preprocessing.py
├── notebooks/
│   ├── 01_eda.ipynb                     # análisis exploratorio
│   └── 02_modelos.ipynb                 # baseline
├── src/
│   └── preprocessing.py                 # carga, limpieza y selección de features
├── results/
│   └── README.md                        # descripción de los resultados del entregable final
└── papers/
    ├── paper_Jair_Aguilera.pdf
    ├── paper_luis_araujo.pdf
    ├── paper_Nilton_Condori.pdf
    └── paper_Pedro_Vilcayauri.pdf
```

## Cómo ejecutar

Los notebooks están preparados para **Google Colab**. Se abren directamente desde estos enlaces (se requiere una cuenta de Google):

- [`01_eda.ipynb`](https://colab.research.google.com/github/jasnai10/AI_Project/blob/main/proyecto/notebooks/01_eda.ipynb): análisis exploratorio.
- [`02_modelos.ipynb`](https://colab.research.google.com/github/jasnai10/AI_Project/blob/main/proyecto/notebooks/02_modelos.ipynb): baseline.

Con el notebook abierto, ejecutar todas las celdas desde **Entorno de ejecución → Ejecutar todas**. La primera celda descarga automáticamente el repositorio en la sesión de Colab, así que no es necesario subir ningún archivo. Se recomienda ejecutar primero `01_eda.ipynb` y luego `02_modelos.ipynb`.

Si los enlaces no funcionan, también se pueden abrir manualmente:

1. Entrar a [Google Colab](https://colab.research.google.com/) y crear un cuaderno con **Nuevo cuaderno**.
2. En el menú superior del cuaderno, ir a **Archivo → Abrir cuaderno** y elegir **GitHub** en el panel izquierdo.
3. Pegar `https://github.com/jasnai10/AI_Project` en el campo de búsqueda, verificar que la rama sea `main` y hacer clic en el notebook en la lista **Ruta**.

Para regenerar `data/processed/cnv_limpio.csv.gz`, cambiar `GUARDAR_PROCESADO = True` en la primera celda de código de `02_modelos.ipynb` antes de ejecutarlo.

## Referencias

- Almeida, A. V. A., Santos, A. O. R., & Nascimento, L. F. C. (2026). Predictive modeling with machine learning for low birth weight in the state of São Paulo: a retrospective population-based study. *São Paulo Medical Journal*. https://doi.org/10.1590/1516-3180.2025.3373.04052026
- Khan, W., Zaki, N., Masud, M. M., Ahmad, A., Ali, L., Ali, N., & Ahmed, L. A. (2022). Infant birth weight estimation and low birth weight classification in United Arab Emirates using machine learning algorithms. *Scientific Reports, 12*, 12110. https://doi.org/10.1038/s41598-022-14393-6
- Liu, Z., Han, N., Su, T., Ji, Y., Bao, H., Zhou, S., Luo, S., Wang, H., Liu, J., & Wang, H.-J. (2022). Interpretable machine learning to identify important predictors of birth weight: A prospective cohort study. *Frontiers in Pediatrics, 10*, 899954. https://doi.org/10.3389/fped.2022.899954
- Ranjbar, A., Montazeri, F., Farashah, M. V., Mehrnoush, V., Darsareh, F., & Roozbeh, N. (2023). Machine learning-based approach for predicting low birth weight. *BMC Pregnancy and Childbirth, 23*, 803. https://doi.org/10.1186/s12884-023-06128-w
