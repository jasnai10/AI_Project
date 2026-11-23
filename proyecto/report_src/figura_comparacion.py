"""Figura 1 del reporte: punto de operación de los cinco modelos e importancia de variables de XGBoost.

Lee las métricas de results/metrics.csv. La importancia por ganancia se copia de la salida del
notebook 06_xgboost.ipynb (sección 4), porque no se guarda en un archivo aparte.
Uso (desde cualquier carpeta): python figura_comparacion.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

AQUI = Path(__file__).resolve().parent
METRICAS = AQUI.parent / "results" / "metrics.csv"
SALIDA = AQUI / "figura_comparacion.png"

# Paleta: un color de énfasis para el modelo elegido y gris para el resto (la identidad la dan las etiquetas)
ACENTO, GRIS = "#2a78d6", "#a3a19b"
TINTA, TINTA_2, EJE, GRILLA = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"

IMPORTANCIA = {  # % de la ganancia total (total_gain), notebook 06, sección 4
    "DUR_EMB_PARTO": 90.83, "Tipo_Parto": 2.83, "departamento": 1.98, "Financiador_Parto": 1.52,
    "Num_embar_madre": 1.02, "Nivel_Instruccion": 0.79, "sexo_nacido": 0.65, "edad_grupo": 0.27,
    "pais_grupo": 0.11,
}

# Nombre en la figura y posición de su etiqueta (recall, precisión). Los tres modelos balanceados
# quedan casi superpuestos: sus etiquetas van a la izquierda, en el mismo orden vertical que los
# puntos, con líneas guía que no se cruzan. Las otras dos van junto a su punto, sin línea guía.
ETIQUETAS = {
    "Baseline: RL simple": ("RL simple (baseline)", (0.487, 0.820), False),
    "RL SMOTE-NC": ("RL con SMOTE-NC", (0.644, 0.618), False),
    "Random Forest balanceado": ("Random forest balanceado", (0.765, 0.405), True),
    "XGBoost balanceado": ("XGBoost balanceado", (0.765, 0.325), True),
    "RL balanceada": ("RL balanceada", (0.765, 0.245), True),
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": EJE, "axes.linewidth": 0.8, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.labelcolor": TINTA_2, "text.color": TINTA,
})

met = pd.read_csv(METRICAS)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.75), gridspec_kw={"width_ratios": [1.12, 1]})

# (a) Punto de operación con el umbral 0.5: recall frente a precisión
for _, fila in met.iterrows():
    elegido = fila["modelo"] == "XGBoost balanceado"
    color = ACENTO if elegido else GRIS
    ax1.scatter(fila["recall"], fila["precision"], s=46, color=color, edgecolor="white",
                linewidth=1.6, zorder=3 if elegido else 2)
    nombre, (lx, ly), guia = ETIQUETAS[fila["modelo"]]
    estilo = dict(fontsize=7.5, color=TINTA if elegido else TINTA_2, va="center",
                  fontweight="semibold" if elegido else "normal")
    if guia:
        ax1.annotate(nombre, xy=(fila["recall"], fila["precision"]), xytext=(lx, ly), ha="right",
                     arrowprops=dict(arrowstyle="-", color=EJE, lw=0.7, relpos=(1, 0.5),
                                     shrinkA=2, shrinkB=4), **estilo)
    else:
        ax1.text(lx, ly, nombre, ha="left", **estilo)
ax1.set(xlim=(0.40, 0.88), ylim=(0.20, 0.92), xlabel="Recall (proporción de casos detectados)",
        ylabel="Precisión")
ax1.set_title("(a) Punto de operación con umbral 0.5 (test)", loc="left", color=TINTA)

# (b) Importancia por ganancia de XGBoost
imp = pd.Series(IMPORTANCIA).sort_values()
ax2.barh(imp.index, imp.values, height=0.55, color=ACENTO)
for i, v in enumerate(imp.values):
    ax2.text(v + 1.5, i, f"{v:.2f}", va="center", fontsize=7.2, color=TINTA_2)
ax2.set(xlim=(0, 108), xlabel="% de la ganancia total")
ax2.set_xticks([0, 25, 50, 75, 100])
ax2.set_title("(b) Importancia de las variables en XGBoost", loc="left", color=TINTA)
ax2.tick_params(axis="y", labelcolor=TINTA)

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRILLA, linewidth=0.6)
    ax.set_axisbelow(True)
ax2.grid(axis="y", visible=False)

fig.tight_layout(w_pad=2.2)
fig.savefig(SALIDA, dpi=300, facecolor="white")
print("Guardado", SALIDA)
