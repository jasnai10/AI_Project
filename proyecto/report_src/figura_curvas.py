"""Figura 2 del reporte: curvas precisión-recall y de calibración de XGBoost balanceado y del baseline (prueba).

Reentrena el modelo elegido en el notebook 06 (mismos datos, partición e hiperparámetros, con los 917 árboles
de su mejor iteración) y el baseline, y comprueba que el AUC-ROC y el AUC-PR coinciden con los del notebook.
Tarda unos 5 minutos. Con --cache <archivo.npz> guarda las predicciones y, si el archivo existe, las reutiliza.
Uso: python figura_curvas.py [--cache ruta.npz]
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent  # proyecto/
SALIDA = AQUI / "figura_curvas.png"
ESPERADO = {"xgb": (0.9130, 0.6818), "base": (0.9081, 0.6774)}  # notebook 06, sección 4 (AUC-ROC, AUC-PR)


def predecir():
    """Ajusta XGBoost balanceado y el baseline con train (2015-2023) y predice el test (2024-2025)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    sys.path.insert(0, str(RAIZ))
    from src.preprocessing import (SEMILLA, cargar_cnv, construir_preprocesador, dividir_temporal, limpiar,
                                   seleccionar_features)

    df_raw, _ = cargar_cnv(ruta_zip=RAIZ / "data" / "raw" / "dataset.zip")
    df = limpiar(df_raw)
    del df_raw
    X, y = seleccionar_features(df, target="bajo_peso")
    X_tr, _, X_te, y_tr, _, y_te = dividir_temporal(X, y, df["anio"])
    del X, y, df

    prep = construir_preprocesador().fit(X_tr)
    ratio = float((y_tr == 0).sum() / y_tr.sum())
    t0 = time.time()
    xgb = XGBClassifier(n_estimators=917, learning_rate=0.1, max_depth=4, min_child_weight=100,
                        scale_pos_weight=ratio, tree_method="hist", n_jobs=-1, random_state=SEMILLA)
    xgb.fit(prep.transform(X_tr), y_tr)
    print(f"XGBoost ajustado en {time.time() - t0:.0f} s")
    base = Pipeline([("prep", construir_preprocesador()),
                     ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEMILLA))])
    base.fit(X_tr, y_tr)
    return y_te.to_numpy(), xgb.predict_proba(prep.transform(X_te))[:, 1], base.predict_proba(X_te)[:, 1]


parser = argparse.ArgumentParser()
parser.add_argument("--cache", type=Path, default=None)
args = parser.parse_args()
if args.cache and args.cache.exists():
    datos = np.load(args.cache)
    y, p_xgb, p_base = datos["y"], datos["xgb"], datos["base"]
else:
    y, p_xgb, p_base = predecir()
    if args.cache:
        np.savez_compressed(args.cache, y=y, xgb=p_xgb, base=p_base)

for clave, p in (("xgb", p_xgb), ("base", p_base)):
    obtenido = (round(roc_auc_score(y, p), 4), round(average_precision_score(y, p), 4))
    estado = "coincide" if obtenido == ESPERADO[clave] else "NO COINCIDE"
    print(f"{clave}: AUC-ROC, AUC-PR = {obtenido} | notebook 06 = {ESPERADO[clave]} -> {estado}")

ACENTO, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_2, EJE, GRILLA = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
    "axes.edgecolor": EJE, "axes.linewidth": 0.8, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.labelcolor": TINTA_2, "text.color": TINTA,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
prevalencia = y.mean()

# (a) Curva precisión-recall, con el punto de operación del umbral 0.5 marcado sobre cada curva
for nombre, p, color, z in (("RL simple (baseline)", p_base, NARANJA, 2), ("XGBoost balanceado", p_xgb, ACENTO, 3)):
    precision, recall, _ = precision_recall_curve(y, p)
    ax1.plot(recall, precision, color=color, lw=1.6, zorder=z,
             label=f"{nombre}: AUC-PR {average_precision_score(y, p):.3f}")
    marcados = p >= 0.5
    r05, p05 = (marcados & (y == 1)).sum() / (y == 1).sum(), y[marcados].mean()
    ax1.scatter(r05, p05, s=34, color=color, edgecolor="white", linewidth=1.4, zorder=z + 2)
    ax1.annotate("umbral 0.5", xy=(r05, p05), xytext=(6, 5), textcoords="offset points", ha="left",
                 va="bottom", fontsize=7, color=TINTA_2)
ax1.axhline(prevalencia, color=EJE, lw=0.8, zorder=1)
ax1.text(0.02, prevalencia + 0.025, f"Azar (prevalencia {prevalencia:.3f})", fontsize=7, color=TINTA_2)
ax1.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precisión")
ax1.set_title("(a) Curva precisión-recall (prueba)", loc="left", color=TINTA)
ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 0.1), frameon=False, handlelength=1.6)

# (b) Calibración por deciles de probabilidad predicha
ax2.plot([0, 1], [0, 1], color=EJE, lw=0.8, zorder=1)
ax2.text(0.62, 0.70, "Calibración perfecta", rotation=38, fontsize=7, color=TINTA_2, ha="center", va="bottom",
         rotation_mode="anchor")
for nombre, p, color, z in (("RL simple (baseline)", p_base, NARANJA, 2), ("XGBoost balanceado", p_xgb, ACENTO, 3)):
    observada, predicha = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax2.plot(predicha, observada, "-o", color=color, lw=1.6, ms=4.6, mec="white", mew=1.0, zorder=z, label=nombre)
ax2.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Probabilidad predicha (media del decil)",
        ylabel="Proporción observada")
ax2.set_title("(b) Calibración por deciles (prueba)", loc="left", color=TINTA)
ax2.legend(loc="upper left", frameon=False, handlelength=1.6)

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRILLA, linewidth=0.6)
    ax.set_axisbelow(True)

fig.tight_layout(w_pad=2.2)
fig.savefig(SALIDA, dpi=300, facecolor="white")
print("Guardado", SALIDA)
