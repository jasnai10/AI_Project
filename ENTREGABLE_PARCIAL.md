# Entregable parcial (semana 8)

## Descripción de lo realizado

Se seleccionó el registro de Certificados de Nacido Vivo (CNV) del MINSA para predecir el bajo peso al nacer en el Perú, y cada integrante revisó un paper sobre el mismo problema con un dataset distinto. Se realizó el análisis exploratorio del dataset, que permitió identificar problemas de calidad de datos y definir las decisiones de preprocesamiento, implementadas en `src/preprocessing.py`. A partir de la literatura revisada se propusieron tres modelos, y se implementó y evaluó el baseline: una regresión logística con y sin balanceo de clases, en un escenario con solo información prenatal y otro que incluye la duración del embarazo.

## Checklist

- [x] `README.md` con la descripción del problema, el objetivo y la metodología.
- [x] Carpeta `papers/` con el resumen del paper revisado por cada integrante:
  - Aguilera, Jair: Almeida et al. (2026)
  - Araujo, Luis: Ranjbar et al. (2023)
  - Condori, Nilton: Liu et al. (2022)
  - Vilcayauri, Pedro: Khan et al. (2022)
- [x] Dataset en `data/raw/` con la descripción de su origen, características y permisos.
- [x] Análisis exploratorio de datos (EDA) en `notebooks/01_eda.ipynb` con visualizaciones.
- [x] Propuesta de modelos identificados en la literatura: regresión logística, random forest y XGBoost.
- [x] Baseline implementado y evaluado en `notebooks/02_modelos.ipynb`: regresión logística con y sin balanceo de clases.
- [x] `ENTREGABLE_PARCIAL.md` con la descripción de lo realizado.
