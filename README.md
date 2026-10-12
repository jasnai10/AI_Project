# Predicción de bajo peso al nacer en el Perú con datos del CNV (MINSA)

**Inteligencia Artificial Aplicada (1INF62) — Pontificia Universidad Católica del Perú, 2026-2**

**Integrantes:** Aguilera, Jair · Araujo, Luis · Condori, Nilton · Vilcayauri, Pedro

## Problema

El bajo peso al nacer (< 2500 g, criterio de la OMS) está asociado a mayor morbilidad y mortalidad neonatal. En el registro de Certificados de Nacido Vivo (CNV) del MINSA, entre 2015 y 2025, afecta al 6.38% de los nacimientos. Identificar con anticipación los embarazos con mayor riesgo permitiría priorizar el seguimiento prenatal, pero los registros administrativos rara vez se han usado con ese fin en el Perú.

## Objetivo

Evaluar modelos de aprendizaje automático para predecir el bajo peso al nacer a partir de los registros del CNV, distinguiendo entre:

- **Escenario A (prenatal):** solo información disponible antes del parto. Es la pregunta central del proyecto.
- **Escenario B (factores asociados):** el Escenario A más la duración del embarazo, que solo se conoce con certeza al nacer. Permite comparar los resultados con la literatura y medir cuánto del desempeño depende de esa variable.

## Dataset

Certificados de Nacido Vivo del MINSA (Plataforma Nacional de Datos Abiertos), corte al 30/11/2025: 4 874 510 registros y 22 variables, de 2015 a 2025. La descripción completa del origen, las variables y los permisos está en `data/raw/`.

## Metodología

1. **Análisis exploratorio** (`notebooks/01_eda.ipynb`): calidad de datos, distribución del target, relación de cada variable con el bajo peso y justificación de las decisiones de preprocesamiento.
2. **Preprocesamiento** (`src/preprocessing.py`): implementa las decisiones del EDA y es compartido por todos los notebooks de modelado. Las principales son:
   - Reparación de 388 filas con el separador `;` dentro del campo de ocupación.
   - Conversión de los códigos de faltante (`IGNORADO`, `-1`, `99`) y de los valores fuera de rango a valores ausentes, con eliminación de casos completos.
   - Exclusión de las variables con fuga de información (talla del recién nacido y variables definidas durante el parto), de `Estado_Civil` (baja validez del registro), de `Hijos_fallec_madre` (código `-1` ambiguo) y de `Hijos_vivo_madre` (redundante con el número de embarazos).
   - Agrupación de categorías poco frecuentes y construcción de `departamento` y `pais_grupo`.
3. **Datos procesados** (`data/processed/cnv_limpio.csv.gz`): resultado de la limpieza, con 4 871 486 registros.
4. **Baseline** (`notebooks/02_modelos.ipynb`): regresión logística con y sin balanceo de clases, siguiendo el diseño de Almeida et al. (2026), evaluada con accuracy, precisión, recall, F1-score y AUC-ROC en ambos escenarios, sobre el mismo conjunto de test (partición 80/20 estratificada).

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

Los tres modelos se entrenarán con y sin balanceo de clases y se evaluarán en los escenarios A y B, sobre el mismo conjunto de test y con las mismas métricas que el baseline.

## Estructura del repositorio

```
proyecto/
├── README.md
├── ENTREGABLE_PARCIAL.md
├── data/
│   ├── raw/            # dataset original (dataset.zip) y su descripción
│   └── processed/      # dataset limpio (cnv_limpio.csv.gz)
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modelos.ipynb
├── src/
│   └── preprocessing.py
├── results/
└── papers/             # un paper por integrante
```

## Cómo ejecutar

Los notebooks están preparados para **Google Colab**:

1. Abrir el notebook desde GitHub en Colab (*Archivo → Abrir cuaderno → GitHub*).
2. Ejecutar todas las celdas en orden (*Entorno de ejecución → Ejecutar todas*). La primera celda clona el repositorio, y `02_modelos.ipynb` importa `src/preprocessing.py` desde ahí.
3. Ejecutar primero `01_eda.ipynb` y luego `02_modelos.ipynb`. Cada uno tarda unos minutos.

Para regenerar `data/processed/cnv_limpio.csv.gz`, cambiar `GUARDAR_PROCESADO = True` en la configuración de `02_modelos.ipynb`.

## Referencias

- Almeida, A. V. A., Santos, A. O. R., & Nascimento, L. F. C. (2026). Predictive modeling with machine learning for low birth weight in the state of São Paulo: a retrospective population-based study. *São Paulo Medical Journal*. https://doi.org/10.1590/1516-3180.2025.3373.04052026
- Khan, W., Zaki, N., Masud, M. M., Ahmad, A., Ali, L., Ali, N., & Ahmed, L. A. (2022). Infant birth weight estimation and low birth weight classification in United Arab Emirates using machine learning algorithms. *Scientific Reports, 12*, 12110. https://doi.org/10.1038/s41598-022-14393-6
- Liu, Z., Han, N., Su, T., Ji, Y., Bao, H., Zhou, S., Luo, S., Wang, H., Liu, J., & Wang, H.-J. (2022). Interpretable machine learning to identify important predictors of birth weight: A prospective cohort study. *Frontiers in Pediatrics, 10*, 899954. https://doi.org/10.3389/fped.2022.899954
- Ranjbar, A., Montazeri, F., Farashah, M. V., Mehrnoush, V., Darsareh, F., & Roozbeh, N. (2023). Machine learning-based approach for predicting low birth weight. *BMC Pregnancy and Childbirth, 23*, 803. https://doi.org/10.1186/s12884-023-06128-w
