# Entregable final (semana 15)

> **Estado:** en progreso. Falta implementar XGBoost, el reporte académico y la presentación.

## Descripción de lo realizado

A partir del análisis exploratorio se definió un único conjunto de 9 variables y una partición temporal: los modelos se entrenan con los nacimientos de 2015–2023 y se evalúan con los de 2024–2025, que simulan su uso en años futuros. Sobre esa base se implementaron cinco modelos propuestos en la literatura revisada, cada uno en su propio notebook y con el mismo protocolo: hiperparámetros ajustados por AUC-PR en validación, umbral de decisión de 0.5 (como en los papers de referencia) y evaluación única en el test con accuracy, precisión, recall, F1, AUC-ROC y AUC-PR. La comparación de resultados y el análisis están en `results/RESULTADOS.md`.

Con cuatro de los cinco modelos evaluados, el Random Forest balanceado es el de mejor capacidad de discriminación (AUC-ROC 0.911, AUC-PR 0.679) y detecta el 79 % de los casos de bajo peso en el test; la conclusión final queda pendiente de XGBoost.

## Organización del repositorio

La profesora indicó que la estructura propuesta en las indicaciones del curso es opcional. El equipo la adaptó así, con su visto bueno:

| Indicación del curso | Cómo se resolvió |
|---|---|
| `notebooks/02_modelos.ipynb` con todos los modelos | Un notebook por modelo: `02_baseline.ipynb`, `03_rl_balanceada.ipynb`, `04_rl_smote.ipynb`, `05_random_forest.ipynb` y `06_xgboost.ipynb` |
| `src/preprocessing.py`, `src/models.py` y `src/evaluation.py` | Solo `src/preprocessing.py`, compartido por todos los notebooks; la implementación y evaluación de cada modelo está en su notebook |
| `results/metrics.csv`, `results/plots/` y `results/model_comparison.txt` | Un solo archivo, `results/RESULTADOS.md`, con la tabla comparativa, los hiperparámetros, el análisis y las limitaciones; los gráficos de cada modelo están en su notebook |
| `requirements.txt` | No se incluye: los notebooks se ejecutan en Google Colab, que ya trae las dependencias (la celda de configuración instala `imbalanced-learn` si falta) |

## Checklist

- [x] Repositorio organizado con `README.md`, datos, notebooks, `src/` y `results/`.
- [x] Dataset limpio regenerado en `data/processed/cnv_limpio.csv.gz` con el preprocesamiento final.
- [x] `src/preprocessing.py` con la carga, limpieza, selección de las 9 features, partición temporal y preprocesador.
- [x] `notebooks/01_eda.ipynb`: análisis exploratorio y justificación de las features.
- [x] `notebooks/02_baseline.ipynb`: regresión logística simple (baseline).
- [x] `notebooks/03_rl_balanceada.ipynb`: regresión logística con `class_weight="balanced"` (Almeida et al., 2026).
- [x] `notebooks/04_rl_smote.ipynb`: regresión logística con sobremuestreo SMOTE-NC (Khan et al., 2022).
- [x] `notebooks/05_random_forest.ipynb`: Random Forest balanceado (Almeida et al., 2026; Ranjbar et al., 2023).
- [ ] `notebooks/06_xgboost.ipynb`: XGBoost (Ranjbar et al., 2023).
- [x] Notebooks ejecutados en Google Colab sin errores y con sus resultados guardados (01 a 05).
- [x] Tabla comparativa de métricas en `results/RESULTADOS.md` (provisional, sin XGBoost).
- [x] Análisis de qué modelo funcionó mejor, por qué, limitaciones e insights en `results/RESULTADOS.md` (provisional).
- [ ] Actualizar `results/RESULTADOS.md` con XGBoost y la conclusión final.
- [ ] Actualizar `README.md` (lista de notebooks, umbral fijo en 0.5, estructura del repositorio y enlaces de Colab).
- [ ] Reporte académico de 3 a 5 páginas en `report.pdf`.
- [ ] Presentación en clase (máximo 5 minutos, enfocada en resultados y conclusiones).
- [ ] `ENTREGABLE_FINAL.md` confirmando la completitud del entregable.
