# Resultados: comparación de modelos

Predicción de bajo peso al nacer (< 2500 g) con el registro de Certificados de Nacido Vivo (CNV) del MINSA.

> **Estado:** resultados de 4 de los 5 modelos. Falta **XGBoost** (`06_xgboost.ipynb`); las tablas y la conclusión se actualizarán cuando esté listo.

---

## 1. Protocolo común

Todos los modelos usan exactamente los mismos datos, la misma partición y las mismas métricas, definidos en `src/preprocessing.py`.

| Aspecto | Decisión |
|---|---|
| Datos | 4 764 960 nacimientos con las 9 features completas (97.81 % de los registros con peso válido) |
| Features (9) | `DUR_EMB_PARTO`, `Num_embar_madre`, `Tipo_Parto`, `sexo_nacido`, `edad_grupo`, `pais_grupo`, `Financiador_Parto`, `Nivel_Instruccion`, `departamento` |
| Preprocesamiento | One-hot para las 7 categóricas y estandarización para las 2 numéricas (53 columnas), ajustado solo con train |
| Partición temporal | Train 2015–2023 (3 237 829), validación 2015–2023 (809 458), test 2024–2025 (717 673) |
| Prevalencia | 6.32 % en train y validación; 6.81 % en test |
| Ajuste de hiperparámetros | Grilla evaluada en **validación** con **AUC-PR**; empate a menos de 0.001, prefiriendo el modelo más simple |
| Umbral de decisión | 0.5, sin calibrar, como en Almeida et al. (2026), Khan et al. (2022) y Ranjbar et al. (2023) |
| Evaluación final | Test 2024–2025, usado una sola vez por modelo |
| Ejecución | Google Colab (entorno estándar de CPU, scikit-learn 1.6.1) |

---

## 2. Tabla comparativa (test 2024–2025)

En negrita, el mejor valor de cada columna. Precisión, recall, F1 y accuracy usan el umbral 0.5.

| Modelo | Accuracy | Recall | F1 | Precisión | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Baseline: regresión logística simple | **0.957** | 0.473 | 0.600 | **0.820** | 0.908 | 0.677 |
| Regresión logística balanceada | 0.865 | **0.809** | 0.449 | 0.310 | 0.908 | 0.677 |
| Regresión logística con SMOTE-NC | 0.948 | 0.630 | **0.624** | 0.618 | 0.907 | 0.677 |
| Random Forest balanceado | 0.880 | 0.793 | 0.475 | 0.339 | **0.911** | **0.679** |
| XGBoost | *pendiente* | | | | | |
| *Clasificador al azar* | — | — | — | *0.068* | *0.500* | *0.068* |

Valores con cuatro decimales:

| Modelo | Accuracy | Recall | F1 | Precisión | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Baseline | 0.9570 | 0.4732 | 0.6002 | 0.8204 | 0.9081 | 0.6774 |
| RL balanceada | 0.8645 | 0.8086 | 0.4485 | 0.3103 | 0.9081 | 0.6769 |
| RL SMOTE-NC | 0.9483 | 0.6296 | 0.6239 | 0.6183 | 0.9071 | 0.6770 |
| Random Forest balanceado | 0.8803 | 0.7933 | 0.4745 | 0.3385 | 0.9107 | 0.6788 |

### Matriz de confusión en test (umbral 0.5)

De los 717 673 nacimientos del test, 48 907 tuvieron bajo peso.

| Modelo | Casos detectados (VP) | Casos perdidos (FN) | Falsas alarmas (FP) | Falsas alarmas por caso detectado |
|---|---|---|---|---|
| Baseline | 23 144 | 25 763 | 5 067 | 0.2 |
| RL balanceada | 39 548 | 9 359 | 87 893 | 2.2 |
| RL SMOTE-NC | 30 790 | 18 117 | 19 006 | 0.6 |
| Random Forest balanceado | 38 800 | 10 107 | 75 828 | 2.0 |

---

## 3. Hiperparámetros elegidos

Todos se eligieron por AUC-PR en validación. Solo se ajustaron hiperparámetros del modelo; los parámetros del algoritmo (solver, paralelismo, semilla) quedaron fijos.

| Modelo | Hiperparámetros ajustados (grilla) | Fijos | Elegido | AUC-PR validación |
|---|---|---|---|---|
| Baseline | — (sin ajuste) | `C=1`, L2, `lbfgs`, sin balanceo | — | 0.6666 |
| RL balanceada | `C` ∈ {1e-5, 1e-4, 1e-3, 0.01, 0.1, 1} × penalización {L2, L1} (12) | `class_weight="balanced"`, `saga` | L2, `C=0.0001` | 0.6655 |
| RL SMOTE-NC | proporción de sobremuestreo {0.25, 0.5} × `C` {1e-4, 1e-3} × penalización {L2, L1} (8) | `k_neighbors=5`, sin `class_weight`, `saga` | proporción 0.25, L2, `C=0.0001` | 0.6659 |
| Random Forest balanceado | `min_samples_leaf` {10, 50, 200} × `max_features` {`"sqrt"`, 0.5} × `max_samples` {0.1, 0.3} (12) | `class_weight="balanced"`, 50 árboles, `max_depth=None` | hojas de 50, `max_features=0.5`, `max_samples=0.3` | **0.6697** |

**Tiempo de ejecución en Colab:** baseline ≈ 13 s de ajuste; RL balanceada ≈ 12 min de grilla; RL SMOTE-NC ≈ 39 min de sobremuestreo + 21 min de grilla; Random Forest ≈ 35 min de grilla.

---

## 4. Criterio para elegir el modelo final

Cada métrica favorece a un modelo distinto (accuracy y precisión al baseline, recall a la RL balanceada, F1 a SMOTE-NC, AUC-ROC y AUC-PR al Random Forest), así que no se elige "por mayoría". Siguiendo las métricas clave de Almeida et al. (2026) y la métrica principal de Ranjbar et al. (2023), se aplica un orden de importancia fijado de antemano:

1. **Capacidad de discriminación (AUC-PR y AUC-ROC):** qué tan bien separa el modelo los casos de bajo peso de los normales, sin depender del umbral. El AUC-PR es la métrica principal porque la clase de interés es rara (6.8 %); es la misma métrica usada para ajustar los hiperparámetros.
2. **Recall:** entre modelos con discriminación equivalente, el que detecta más casos, por ser la prioridad clínica.
3. **F1 y precisión:** se reportan para describir el costo en falsas alarmas.

La accuracy no se usa como criterio: con 6.8 % de prevalencia, predecir siempre "peso normal" ya daría cerca de 93 %.

---

## 5. Resultado provisional

**Modelo con mejor desempeño hasta ahora: Random Forest balanceado** (a reserva de XGBoost).

- Es el único que supera al baseline en las dos métricas de discriminación: AUC-ROC 0.911 frente a 0.908 y AUC-PR 0.679 frente a 0.677. La ventaja es pequeña (0.003 y 0.001).
- Con el umbral 0.5 detecta el 79 % de los recién nacidos con bajo peso (38 800 de 48 907), con una precisión de 0.34: una de cada tres alertas es un caso real, cinco veces más que al azar.
- Frente a la regresión logística balanceada, que tiene un recall parecido (0.81), da unas 12 000 falsas alarmas menos.

---

## 6. Análisis

**1. Todos los modelos discriminan de forma parecida.** El AUC-PR va de 0.677 a 0.679 y el AUC-ROC de 0.907 a 0.911: unas 10 veces el azar en AUC-PR. Ni el tratamiento del desbalance ni un modelo no lineal cambian mucho la capacidad de separar las clases con estas 9 variables.

**2. El balanceo cambia el punto de operación, no la discriminación.** La regresión logística balanceada y la de SMOTE-NC tienen el mismo AUC-PR que el baseline; lo que cambia es cuántos casos marcan con el umbral 0.5. El baseline marca pocos (recall 0.47, precisión 0.82), la balanceada marca muchos (recall 0.81, precisión 0.31) y SMOTE-NC con proporción 0.25 queda en el medio (recall 0.63, precisión 0.62, el mejor F1). Es el mismo patrón que reporta Almeida et al.: balancear sube el recall y baja la precisión, con el AUC-ROC prácticamente igual.

**3. El Random Forest aporta una mejora pequeña.** La flexibilidad de los árboles (relaciones no lineales e interacciones) mejora levemente la discriminación y reduce las falsas alarmas para un mismo nivel de recall, a cambio de un tiempo de entrenamiento mucho mayor. En su grilla, `max_features` fue el hiperparámetro más influyente (`"sqrt"` rindió claramente peor) y las hojas de 10 registros mostraron sobreajuste.

**4. SMOTE-NC no mejora la discriminación.** A diferencia de Khan et al. (2022), que trabajó con 89 casos de bajo peso, aquí hay más de 200 000 casos reales en train; los casos sintéticos no aportan información nueva. Además, fue el modelo más costoso de entrenar (≈ 1 hora en Colab).

**5. Calibración.** Solo el baseline produce probabilidades bien calibradas. Los modelos balanceados (RL balanceada, SMOTE-NC y Random Forest) sobreestiman el riesgo: en el decil de mayor riesgo predicen entre 0.68 y 0.88, cuando la proporción observada es cercana a 0.48. Para usar sus probabilidades como riesgo absoluto habría que recalibrarlas.

**6. Estabilidad en el tiempo.** El baseline mantiene su desempeño entre validación (2015–2023) y test (2024–2025): AUC-ROC de 0.904 a 0.908. Los modelos entrenados con el pasado generalizan a los años siguientes.

### Comparación con la literatura

| Estudio | Mejor modelo | AUC-ROC | Recall | Prevalencia |
|---|---|---|---|---|
| Almeida et al. (2026) | RL balanceada | 0.85 | 0.688 | 9.5 % |
| Ranjbar et al. (2023) | XGBoost | 0.79 | 0.69 | 14.5 % |
| Este trabajo (provisional) | Random Forest balanceado | 0.911 | 0.793 | 6.8 % |

Las cifras no son directamente comparables (otros países, variables, prevalencias y particiones), pero el desempeño está en el rango de la literatura. A diferencia de Almeida, donde el random forest rindió peor que la regresión logística (AUC-ROC 0.78–0.81 frente a 0.85), aquí queda a su nivel o levemente por encima.

---

## 7. Limitaciones

- **`DUR_EMB_PARTO` se conoce en el momento del parto.** Los modelos sirven para anticipar, antes del pesaje, qué recién nacidos pueden necesitar atención especial; no son una predicción prenatal.
- **Umbral fijo en 0.5**, como en los papers de referencia; precisión, recall y F1 dependen de esa elección.
- **El modelo final se elige comparando métricas en el test,** como en los estudios de referencia; esto puede dar una estimación algo optimista del modelo elegido.
- **La prevalencia del test (6.81 %) es mayor que la de train (6.32 %),** lo que cambia el nivel de la precisión y el F1.
- **El CNV no registra variables clínicas** (controles prenatales, anemia, hipertensión, IMC) que podrían mejorar la predicción.
- **Ajuste acotado por cómputo:** una sola partición de validación y una sola semilla; en SMOTE-NC no se ajustó `k_neighbors` ni se probó el balance total, y el Random Forest usó 50 árboles.
- **Redondeo del peso en 2500 g** (16 353 registros exactos), que introduce ruido en la etiqueta justo en el umbral.
- **El test no incluye diciembre de 2025.**

---

## 8. Pendiente

- [ ] `06_xgboost.ipynb`: implementar, ajustar y evaluar con el mismo protocolo.
- [ ] Actualizar las tablas, el modelo final y el análisis con los resultados de XGBoost.
