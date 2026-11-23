"""Figuras del reporte basadas en las predicciones de prueba, y análisis de errores de XGBoost.

Genera, a ancho de una columna del formato del curso:
  - figura_precision_recall.png: curvas precisión-recall de XGBoost balanceado y del baseline.
  - figura_calibracion.png: curvas de calibración por deciles de ambos modelos.
e imprime el análisis de errores que cita la sección VI (casos perdidos y falsas alarmas por edad gestacional).

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
    return {"y": y_te.to_numpy(), "xgb": xgb.predict_proba(prep.transform(X_te))[:, 1],
            "base": base.predict_proba(X_te)[:, 1], "semanas": X_te["DUR_EMB_PARTO"].to_numpy(dtype=float),
            "multiple": (X_te["Tipo_Parto"].astype(str) == "MULTIPLE").to_numpy()}


parser = argparse.ArgumentParser()
parser.add_argument("--cache", type=Path, default=None)
args = parser.parse_args()
if args.cache and args.cache.exists() and "semanas" in np.load(args.cache).files:
    d = dict(np.load(args.cache))
else:
    d = predecir()
    if args.cache:
        np.savez_compressed(args.cache, **d)
y, p_xgb, p_base = d["y"], d["xgb"], d["base"]

for clave, p in (("xgb", p_xgb), ("base", p_base)):
    obtenido = (round(roc_auc_score(y, p), 4), round(average_precision_score(y, p), 4))
    estado = "coincide" if obtenido == ESPERADO[clave] else "NO COINCIDE"
    print(f"{clave}: AUC-ROC, AUC-PR = {obtenido} | notebook 06 = {ESPERADO[clave]} -> {estado}")

# Análisis de errores con el umbral 0.5 (sección VI del reporte)
pre = d["semanas"] < 37
for clave, p in (("xgb", p_xgb), ("base", p_base)):
    marca = p >= 0.5
    vp, fn, fp = marca & (y == 1), ~marca & (y == 1), marca & (y == 0)
    print(f"\n[{clave}] VP={vp.sum():,} FN={fn.sum():,} FP={fp.sum():,}")
    for nombre, grupo in (("pretérmino (<37 sem)", pre), ("a término (>=37 sem)", ~pre)):
        casos = (y == 1) & grupo
        print(f"  {nombre}: casos={casos.sum():,} | recall={vp[grupo].sum() / casos.sum():.3f} | "
              f"FN={fn[grupo].sum():,} ({fn[grupo].sum() / fn.sum():.1%} de los FN) | "
              f"FP={fp[grupo].sum():,} ({fp[grupo].sum() / fp.sum():.1%} de las FP)")
    print(f"  partos múltiples: recall={vp[d['multiple']].sum() / ((y == 1) & d['multiple']).sum():.3f}")
print(f"\nCasos de bajo peso a término: {((y == 1) & ~pre).sum() / (y == 1).sum():.1%} del total de casos")

# Figuras a ancho de columna, con la tipografía del reporte
ACENTO, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_2, EJE, GRILLA = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"
plt.rcParams.update({
    "font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": EJE, "axes.linewidth": 0.8, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.labelcolor": TINTA, "text.color": TINTA,
})
MODELOS = (("RL simple (baseline)", p_base, NARANJA, 2), ("XGBoost balanceado", p_xgb, ACENTO, 3))


def preparar(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRILLA, linewidth=0.6)
    ax.set_axisbelow(True)


# Curvas precisión-recall, con el punto de operación del umbral 0.5 sobre cada curva
fig, ax = plt.subplots(figsize=(3.35, 2.35))
for nombre, p, color, z in MODELOS:
    precision, recall, _ = precision_recall_curve(y, p)
    ax.plot(recall, precision, color=color, lw=1.5, zorder=z,
            label=f"{nombre}: AUC-PR {average_precision_score(y, p):.3f}")
    marca = p >= 0.5
    r05, p05 = (marca & (y == 1)).sum() / (y == 1).sum(), y[marca].mean()
    ax.scatter(r05, p05, s=30, color=color, edgecolor="white", linewidth=1.3, zorder=z + 2)
    ax.annotate("umbral 0.5", xy=(r05, p05), xytext=(5, 4), textcoords="offset points", ha="left",
                va="bottom", fontsize=7, color=TINTA_2)
ax.axhline(y.mean(), color=EJE, lw=0.8, zorder=1)
ax.text(0.02, y.mean() + 0.025, f"Azar (prevalencia {y.mean():.3f})", fontsize=7, color=TINTA_2)
ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Recall", ylabel="Precisión")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.11), frameon=False, handlelength=1.5)
preparar(ax)
fig.tight_layout(pad=0.3)
fig.savefig(AQUI / "figura_precision_recall.png", dpi=300, facecolor="white")
plt.close(fig)

# Calibración por deciles de probabilidad predicha
fig, ax = plt.subplots(figsize=(3.35, 2.35))
ax.plot([0, 1], [0, 1], color=EJE, lw=0.8, zorder=1)
ax.text(0.63, 0.71, "Calibración perfecta", rotation=35, fontsize=7, color=TINTA_2, ha="center", va="bottom",
        rotation_mode="anchor")
for nombre, p, color, z in MODELOS:
    observada, predicha = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot(predicha, observada, "-o", color=color, lw=1.5, ms=4.2, mec="white", mew=0.9, zorder=z, label=nombre)
ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Probabilidad predicha (media del decil)",
       ylabel="Proporción observada")
ax.legend(loc="upper left", frameon=False, handlelength=1.5)
preparar(ax)
fig.tight_layout(pad=0.3)
fig.savefig(AQUI / "figura_calibracion.png", dpi=300, facecolor="white")
plt.close(fig)
print("\nGuardadas figura_precision_recall.png y figura_calibracion.png en", AQUI)
