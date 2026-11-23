"""Figuras del reporte que no requieren reentrenar modelos, a ancho de una columna del formato del curso.

  - figura_puntos_operacion.png: precisión y recall de los cinco modelos con el umbral 0.5 (results/metrics.csv).
  - figura_importancia.png: importancia por ganancia de las variables en XGBoost balanceado. Los valores se copian
    de la salida del notebook 06_xgboost.ipynb (sección 4), porque no se guardan en un archivo aparte.
Uso (desde cualquier carpeta): python figura_comparacion.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

AQUI = Path(__file__).resolve().parent
METRICAS = AQUI.parent / "results" / "metrics.csv"

# Un color de énfasis para el modelo elegido y gris para el resto (la identidad la dan las etiquetas)
ACENTO, GRIS = "#2a78d6", "#a3a19b"
TINTA, TINTA_2, EJE, GRILLA = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"

IMPORTANCIA = {  # % de la ganancia total (total_gain), notebook 06, sección 4
    "DUR_EMB_PARTO": 90.83, "Tipo_Parto": 2.83, "departamento": 1.98, "Financiador_Parto": 1.52,
    "Num_embar_madre": 1.02, "Nivel_Instruccion": 0.79, "sexo_nacido": 0.65, "edad_grupo": 0.27,
    "pais_grupo": 0.11,
}

# Nombre en la figura, posición de la etiqueta (recall, precisión) y si lleva línea guía. Los tres modelos
# balanceados quedan casi superpuestos: sus etiquetas van a la izquierda, en el mismo orden vertical que los
# puntos, con líneas guía que no se cruzan.
ETIQUETAS = {
    "Baseline: RL simple": ("RL simple (baseline)", (0.488, 0.820), False),
    "RL SMOTE-NC": ("RL con SMOTE-NC", (0.645, 0.618), False),
    "Random Forest balanceado": ("Random forest balanceado", (0.765, 0.415), True),
    "XGBoost balanceado": ("XGBoost balanceado", (0.765, 0.325), True),
    "RL balanceada": ("RL balanceada", (0.765, 0.235), True),
}

plt.rcParams.update({
    "font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": EJE, "axes.linewidth": 0.8, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.labelcolor": TINTA, "text.color": TINTA,
})


def preparar(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRILLA, linewidth=0.6)
    ax.set_axisbelow(True)


# Punto de operación con el umbral 0.5: recall frente a precisión
met = pd.read_csv(METRICAS)
fig, ax = plt.subplots(figsize=(3.35, 2.35))
for _, fila in met.iterrows():
    elegido = fila["modelo"] == "XGBoost balanceado"
    ax.scatter(fila["recall"], fila["precision"], s=40, color=ACENTO if elegido else GRIS, edgecolor="white",
               linewidth=1.5, zorder=3 if elegido else 2)
    nombre, (lx, ly), guia = ETIQUETAS[fila["modelo"]]
    estilo = dict(fontsize=7.5, color=TINTA if elegido else TINTA_2, va="center",
                  fontweight="bold" if elegido else "normal")
    if guia:
        ax.annotate(nombre, xy=(fila["recall"], fila["precision"]), xytext=(lx, ly), ha="right",
                    arrowprops=dict(arrowstyle="-", color=EJE, lw=0.7, relpos=(1, 0.5), shrinkA=2, shrinkB=4),
                    **estilo)
    else:
        ax.text(lx, ly, nombre, ha="left", **estilo)
ax.set(xlim=(0.40, 0.88), ylim=(0.18, 0.92), xlabel="Recall (proporción de casos detectados)",
       ylabel="Precisión")
preparar(ax)
fig.tight_layout(pad=0.3)
fig.savefig(AQUI / "figura_puntos_operacion.png", dpi=300, facecolor="white")
plt.close(fig)

# Importancia por ganancia de XGBoost (una sola serie: un solo color, valores al final de cada barra)
imp = pd.Series(IMPORTANCIA).sort_values()
fig, ax = plt.subplots(figsize=(3.35, 2.15))
ax.barh(imp.index, imp.values, height=0.55, color=ACENTO)
for i, v in enumerate(imp.values):
    ax.text(v + 1.5, i, f"{v:.2f}", va="center", fontsize=7.2, color=TINTA_2)
ax.set(xlim=(0, 110), xlabel="% de la ganancia total")
ax.set_xticks([0, 25, 50, 75, 100])
ax.tick_params(axis="y", labelcolor=TINTA)
preparar(ax)
ax.grid(axis="y", visible=False)
fig.tight_layout(pad=0.3)
fig.savefig(AQUI / "figura_importancia.png", dpi=300, facecolor="white")
plt.close(fig)
print("Guardadas figura_puntos_operacion.png y figura_importancia.png en", AQUI)
