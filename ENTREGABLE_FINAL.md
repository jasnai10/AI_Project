# Entregable final (semana 15)

> **Estado:** en progreso. Los cinco modelos están implementados y comparados, y el reporte académico está listo; falta la presentación.

## Descripción de lo realizado

A partir del análisis exploratorio se definió un único conjunto de 9 variables y una partición temporal: los modelos se entrenan con los nacimientos de 2015–2023 y se evalúan con los de 2024–2025, que simulan su uso en años futuros. Sobre esa base se implementaron cinco modelos propuestos en la literatura revisada, cada uno en su propio notebook y con el mismo protocolo: hiperparámetros ajustados por AUC-PR en validación, umbral de decisión de 0.5 (como en los papers de referencia) y evaluación única en el test con accuracy, precisión, recall, F1, AUC-ROC y AUC-PR. La comparación de resultados y el análisis están en `results/model_comparison.txt`.

De los cinco modelos, XGBoost balanceado es el de mejor capacidad de discriminación (AUC-ROC 0.913, AUC-PR 0.682) y detecta el 80 % de los casos de bajo peso en el test. Su ventaja sobre el baseline es pequeña (0.005 en AUC-ROC y 0.004 en AUC-PR): los cinco modelos discriminan de forma parecida, y lo que más cambia entre ellos es el equilibrio entre casos detectados y falsas alarmas.

## Organización del repositorio

La profesora indicó que la estructura propuesta en las indicaciones del curso es opcional. El equipo la adaptó así, con su visto bueno:

| Indicación del curso | Cómo se resolvió |
|---|---|
| `notebooks/02_modelos.ipynb` con todos los modelos | Un notebook por modelo: `02_baseline.ipynb`, `03_rl_balanceada.ipynb`, `04_rl_smote.ipynb`, `05_random_forest.ipynb` y `06_xgboost.ipynb` |
| `src/preprocessing.py`, `src/models.py` y `src/evaluation.py` | Solo `src/preprocessing.py`, compartido por todos los notebooks; la implementación y evaluación de cada modelo está en su notebook |
| `requirements.txt` | No se incluye: los notebooks se ejecutan en Google Colab, que ya trae las dependencias (la celda de configuración instala `imbalanced-learn` o `xgboost` si faltan) |

## Checklist

- [x] Repositorio organizado con `README.md`, datos, notebooks, `src/` y `results/`.
- [x] Dataset limpio regenerado en `data/processed/cnv_limpio.csv.gz` con el preprocesamiento final.
- [x] `src/preprocessing.py` con la carga, limpieza, selección de las 9 features, partición temporal y preprocesador.
- [x] `notebooks/01_eda.ipynb`: análisis exploratorio y justificación de las features.
- [x] `notebooks/02_baseline.ipynb`: regresión logística simple (baseline).
- [x] `notebooks/03_rl_balanceada.ipynb`: regresión logística con `class_weight="balanced"` (Almeida et al., 2026).
- [x] `notebooks/04_rl_smote.ipynb`: regresión logística con sobremuestreo SMOTE-NC (Khan et al., 2022).
- [x] `notebooks/05_random_forest.ipynb`: Random Forest balanceado (Almeida et al., 2026; Ranjbar et al., 2023).
- [x] `notebooks/06_xgboost.ipynb`: XGBoost (Ranjbar et al., 2023).
- [x] Notebooks ejecutados sin errores y con sus resultados guardados: 01 a 05 en Google Colab y 06 en local.
- [x] Tabla comparativa con las 6 métricas en test en `results/metrics.csv` y en `results/model_comparison.txt`.
- [x] Gráficos de cada modelo en `results/plots/`.
- [x] Análisis de qué modelo funcionó mejor, por qué, limitaciones e insights en `results/model_comparison.txt`.
- [x] `README.md` actualizado: metodología con umbral fijo en 0.5, modelos implementados, resultados, estructura del repositorio y enlaces de Colab de cada notebook.
- [x] Actualizar `results/metrics.csv`, `results/plots/`, `results/model_comparison.txt` y `README.md` con XGBoost y la conclusión final.
- [x] Reporte académico de 5 páginas en `proyecto/report.pdf`, con la declaración de uso de IA y la contribución de cada integrante.
- [ ] Presentación en clase (máximo 5 minutos, enfocada en resultados y conclusiones).
- [ ] `ENTREGABLE_FINAL.md` confirmando la completitud del entregable.
