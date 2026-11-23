"""
Pruebas del modelo XGBoost de notebooks/06_xgboost.ipynb.

Comprueban que el modelo funciona con el preprocesamiento del proyecto (src/preprocessing.py):

1. Pruebas del modelo, rápidas, con un CNV sintético de relaciones conocidas. Usan las funciones
   modelo_xgboost() y evaluar() tomadas del propio notebook, así que prueban exactamente el modelo
   que se entrena ahí:
   - entrena y devuelve probabilidades válidas;
   - aprende la relación con la duración del embarazo (buen AUC en datos que no vio);
   - el early stopping elige el número de árboles por AUC-PR en validación;
   - el balanceo (scale_pos_weight) sube el recall sin cambiar la capacidad de discriminar;
   - capta una interacción entre variables que la regresión logística no puede representar;
   - da las mismas predicciones con la matriz dispersa del preprocesador que con la densa;
   - es reproducible y tolera categorías que no aparecieron en train.
2. Prueba de integración: ejecuta el notebook completo, de principio a fin, con el CNV sintético
   en lugar del real, y revisa la grilla, el modelo elegido, las métricas en test y los gráficos.
3. Prueba con el CNV real (lenta, solo con --lento): entrena el modelo del notebook con una muestra
   de train y comprueba en validación que discrimina al menos como el baseline. No usa el test.

Cómo ejecutarlas (desde la carpeta AI_Project):
    pip install xgboost scikit-learn pandas matplotlib pytest nbformat nbclient ipykernel
    python -m pytest proyecto/tests -v              # pruebas 1 y 2 (alrededor de 1 minuto)
    python -m pytest proyecto/tests -v --lento      # además, la prueba 3 con el CNV real
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

pytest.importorskip("xgboost")
from xgboost import XGBClassifier  # noqa: E402

PROYECTO = Path(__file__).resolve().parents[1]
NOTEBOOK = PROYECTO / "notebooks" / "06_xgboost.ipynb"
TESTS = Path(__file__).resolve().parent


def ajustar(nb, d, max_depth=4, min_child_weight=1, scale_pos_weight=1.0, X_tr=None, X_val=None):
    """Entrena el modelo del notebook con early stopping en validación, igual que en la grilla."""
    X_tr = d.Xtr_t if X_tr is None else X_tr
    X_val = d.Xval_t if X_val is None else X_val
    modelo = nb.modelo_xgboost(max_depth, min_child_weight, scale_pos_weight)
    return modelo.fit(X_tr, d.y_tr, eval_set=[(X_val, d.y_val)], verbose=False)


@pytest.fixture(scope="module")
def modelo(nb, datos):
    return ajustar(nb, datos)


# ---------------------------------------------------------------------------
# 1. Pruebas del modelo (CNV sintético)
# ---------------------------------------------------------------------------

def test_el_modelo_del_notebook_usa_aucpr_y_early_stopping(nb):
    m = nb.modelo_xgboost(6, 100, 14.8)
    assert isinstance(m, XGBClassifier)
    params = m.get_params()
    assert (params["max_depth"], params["min_child_weight"], params["scale_pos_weight"]) == (6, 100, 14.8)
    assert params["eval_metric"] == "aucpr"          # el número de árboles se elige por AUC-PR, como la grilla
    assert params["early_stopping_rounds"] == 50
    assert params["n_estimators"] == nb.MAX_ARBOLES
    assert nb.UMBRAL == 0.5                           # umbral fijo del protocolo


def test_entrena_y_devuelve_probabilidades_validas(modelo, datos):
    proba = modelo.predict_proba(datos.Xte_t)
    assert proba.shape == (datos.Xte_t.shape[0], 2)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-6)
    assert set(np.unique(modelo.predict(datos.Xte_t))) == {0, 1}


def test_aprende_la_senal_en_datos_que_no_vio(modelo, datos, nb):
    p = modelo.predict_proba(datos.Xte_t)[:, 1]
    m = nb.evaluar(datos.y_te, p)
    assert m["auc_roc"] > 0.85
    assert m["auc_pr"] > 3 * datos.y_te.mean()       # muy por encima del azar (la prevalencia)
    # el riesgo predicho cae con la duración del embarazo, como en los datos simulados
    semanas = datos.X_te["DUR_EMB_PARTO"].to_numpy()
    assert p[semanas <= 35].mean() > 0.4
    assert p[semanas >= 39].mean() < 0.1


def test_early_stopping_se_queda_con_la_mejor_iteracion(modelo, datos, nb):
    mejor = modelo.best_iteration
    total = modelo.get_booster().num_boosted_rounds()
    assert mejor + 1 < total < nb.MAX_ARBOLES        # se detuvo antes del límite de árboles
    historia = modelo.evals_result()["validation_0"]["aucpr"]
    assert int(np.argmax(historia)) == mejor         # la mejor iteración es la de mayor AUC-PR en validación
    assert historia[mejor] == pytest.approx(modelo.best_score)
    # al predecir usa solo los árboles hasta la mejor iteración, no los de después
    X = datos.Xval_t
    p = modelo.predict_proba(X)[:, 1]
    np.testing.assert_allclose(p, modelo.predict_proba(X, iteration_range=(0, mejor + 1))[:, 1])
    assert not np.allclose(p, modelo.predict_proba(X, iteration_range=(0, total))[:, 1])


def test_el_balanceo_sube_el_recall_sin_cambiar_la_discriminacion(nb, datos):
    sin = ajustar(nb, datos, scale_pos_weight=1.0)
    con = ajustar(nb, datos, scale_pos_weight=datos.ratio)
    p_sin, p_con = sin.predict_proba(datos.Xte_t)[:, 1], con.predict_proba(datos.Xte_t)[:, 1]
    m_sin, m_con = nb.evaluar(datos.y_te, p_sin), nb.evaluar(datos.y_te, p_con)
    assert m_con["recall"] > m_sin["recall"] + 0.15        # con el umbral 0.5 detecta muchos más casos
    assert m_con["precision"] < m_sin["precision"]          # a costa de más falsas alarmas
    assert abs(m_con["auc_roc"] - m_sin["auc_roc"]) < 0.02  # pero ordena a los recién nacidos casi igual
    assert p_con.mean() > 2 * p_sin.mean()                  # y sus probabilidades quedan infladas


def test_capta_interacciones_que_la_regresion_logistica_no_puede(nb, datos_con_interaccion):
    d = datos_con_interaccion
    xgb = ajustar(nb, d)
    baseline = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(d.Xtr_t, d.y_tr)
    auc_xgb = roc_auc_score(d.y_te, xgb.predict_proba(d.Xte_t)[:, 1])
    auc_rl = roc_auc_score(d.y_te, baseline.predict_proba(d.Xte_t)[:, 1])
    assert auc_xgb > auc_rl + 0.03


def test_matriz_dispersa_y_densa_dan_las_mismas_predicciones(nb, datos):
    assert sp.issparse(datos.Xtr_t)                  # el preprocesador del proyecto devuelve una matriz dispersa
    dispersa = ajustar(nb, datos)
    densa = ajustar(nb, datos, X_tr=datos.Xtr_t.toarray(), X_val=datos.Xval_t.toarray())
    assert dispersa.best_iteration == densa.best_iteration
    np.testing.assert_allclose(dispersa.predict_proba(datos.Xte_t)[:, 1],
                               densa.predict_proba(datos.Xte_t.toarray())[:, 1], atol=1e-6)


def test_es_reproducible(nb, datos, modelo):
    otro = ajustar(nb, datos)
    np.testing.assert_array_equal(modelo.predict_proba(datos.Xte_t), otro.predict_proba(datos.Xte_t))


def test_tolera_categorias_que_no_estaban_en_train(datos, modelo):
    final = Pipeline([("prep", datos.prep), ("clf", modelo)])     # como el modelo final del notebook
    X = datos.X_te.copy()
    X["departamento"] = X["departamento"].cat.add_categories(["Región nueva"])
    X.iloc[:50, X.columns.get_loc("departamento")] = "Región nueva"
    p = final.predict_proba(X)[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


# ---------------------------------------------------------------------------
# 2. Integración: el notebook completo con el CNV sintético
# ---------------------------------------------------------------------------

CELDA_DATOS = '''
# Celda agregada por la prueba: reemplaza el CNV real por uno sintético
import sys
sys.path.insert(0, {tests!r})
from datos_sinteticos import cnv_crudo_sintetico
def cargar_cnv(*args, **kwargs):
    return cnv_crudo_sintetico(n={n}, semilla={semilla}), {{}}
'''

CELDA_RESUMEN = '''
# Celda agregada por la prueba: guarda un resumen de los resultados del notebook
import json
with open({salida!r}, "w", encoding="utf-8") as f:
    json.dump({{"grilla": res.to_dict(orient="records"),
               "elegido": {{"max_depth": int(mejor["max_depth"]), "min_child_weight": int(mejor["min_child_weight"]),
                           "balanceo": str(mejor["balanceo"]), "arboles": int(mejor["arboles"])}},
               "nombre": NOMBRE,
               "metricas": metricas.to_dict(orient="index"),
               "prevalencia_val": float(y_val.mean()),
               "prevalencia_test": float(y_te.mean()),
               "importancia": imp.to_dict()}}, f, ensure_ascii=False, default=float)
'''


@pytest.fixture(scope="module")
def notebook_ejecutado(tmp_path_factory):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client.manager import KernelManager

    salida = tmp_path_factory.mktemp("notebook_06") / "resumen.json"
    cuaderno = nbformat.read(NOTEBOOK, as_version=4)
    i = next(k for k, c in enumerate(cuaderno.cells)
             if c.cell_type == "code" and "from src.preprocessing import" in c.source)
    cuaderno.cells.insert(i + 1, nbformat.v4.new_code_cell(CELDA_DATOS.format(tests=str(TESTS), n=20_000, semilla=3)))
    cuaderno.cells.append(nbformat.v4.new_code_cell(CELDA_RESUMEN.format(salida=str(salida))))

    # Kernel con el mismo intérprete que ejecuta pytest (ignora kernels instalados con otro Python)
    especificaciones = KernelSpecManager()
    especificaciones.kernel_dirs = []
    km = KernelManager(kernel_name="python3", kernel_spec_manager=especificaciones)
    cliente = nbclient.NotebookClient(cuaderno, km=km, timeout=900,
                                      resources={"metadata": {"path": str(NOTEBOOK.parent)}})
    cliente.execute()        # lanza CellExecutionError si alguna celda falla
    return cuaderno, json.loads(salida.read_text(encoding="utf-8"))


def test_notebook_se_ejecuta_sin_errores_y_dibuja_los_graficos(notebook_ejecutado):
    cuaderno, _ = notebook_ejecutado
    salidas = [o for c in cuaderno.cells if c.cell_type == "code" for o in c.get("outputs", [])]
    assert not [o for o in salidas if o.output_type == "error"]
    graficos = sum("image/png" in o.get("data", {}) for o in salidas)
    assert graficos == 3          # grilla train vs validación, curvas en test e importancia de variables


def test_notebook_recorre_la_grilla_y_elige_un_modelo_empatado_con_el_mejor(notebook_ejecutado, nb):
    _, r = notebook_ejecutado
    grilla = pd.DataFrame(r["grilla"])
    combinaciones = ["max_depth", "min_child_weight", "balanceo"]
    assert len(grilla) == 12 and not grilla.duplicated(combinaciones).any()
    assert (grilla["arboles"] <= nb.MAX_ARBOLES - 50).all()      # el early stopping se activó en todos
    assert (grilla["auc_roc_val"] > 0.5).all()                      # todos mejores que el azar...
    assert (grilla["auc_pr_val"] > 2 * r["prevalencia_val"]).all()  # ...cuyo AUC-PR es la prevalencia
    e = r["elegido"]
    fila = grilla[(grilla.max_depth == e["max_depth"]) & (grilla.min_child_weight == e["min_child_weight"])
                  & (grilla.balanceo == e["balanceo"])]
    assert len(fila) == 1 and int(fila["arboles"].iloc[0]) == e["arboles"]
    assert fila["auc_pr_val"].iloc[0] >= grilla["auc_pr_val"].max() - 0.001     # criterio de selección
    assert r["nombre"] == ("XGBoost balanceado" if e["balanceo"] == "sí" else "XGBoost")


def test_notebook_evalua_en_test_y_supera_ampliamente_al_azar(notebook_ejecutado):
    _, r = notebook_ejecutado
    assert set(r["metricas"]) == {r["nombre"], "RL simple (baseline)"}
    m = r["metricas"][r["nombre"]]
    assert set(m) == {"auc_roc", "auc_pr", "precision", "recall", "f1", "accuracy"}
    assert all(0 <= v <= 1 for v in m.values())
    assert m["auc_roc"] > 0.85
    assert m["auc_pr"] > 3 * r["prevalencia_test"]
    # la variable más usada es la duración del embarazo, la que domina en los datos simulados
    assert max(r["importancia"], key=r["importancia"].get) == "DUR_EMB_PARTO"
    assert sum(r["importancia"].values()) == pytest.approx(100)


# ---------------------------------------------------------------------------
# 3. CNV real (lenta: solo con --lento)
# ---------------------------------------------------------------------------

@pytest.mark.lento
def test_xgboost_con_el_cnv_real_discrimina_al_menos_como_el_baseline(nb):
    from src.preprocessing import (cargar_cnv, construir_preprocesador, dividir_temporal, limpiar,
                                   seleccionar_features)

    df_raw, _ = cargar_cnv(ruta_zip=PROYECTO / "data" / "raw" / "dataset.zip", verbose=False)
    df = limpiar(df_raw, verbose=False)
    del df_raw
    X, y = seleccionar_features(df, target="bajo_peso", verbose=False)
    X_tr, X_val, _, y_tr, y_val, _ = dividir_temporal(X, y, df["anio"])     # el test no se usa
    del X, y, df
    X_tr = X_tr.sample(400_000, random_state=0)     # muestra de train para que la prueba tarde pocos minutos
    y_tr = y_tr.loc[X_tr.index]

    prep = construir_preprocesador().fit(X_tr)
    Xtr_t, Xval_t = prep.transform(X_tr), prep.transform(X_val)
    xgb = nb.modelo_xgboost(6, 1, 1.0).fit(Xtr_t, y_tr, eval_set=[(Xval_t, y_val)], verbose=False)
    baseline = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(Xtr_t, y_tr)
    m_xgb = nb.evaluar(y_val, xgb.predict_proba(Xval_t)[:, 1])
    m_rl = nb.evaluar(y_val, baseline.predict_proba(Xval_t)[:, 1])
    print(f"\nValidación 2015-2023: XGBoost AUC-ROC {m_xgb['auc_roc']:.4f}, AUC-PR {m_xgb['auc_pr']:.4f} "
          f"({xgb.best_iteration + 1} árboles) | baseline AUC-ROC {m_rl['auc_roc']:.4f}, AUC-PR {m_rl['auc_pr']:.4f}")
    assert m_xgb["auc_roc"] > 0.90
    assert m_xgb["auc_pr"] > 0.65                    # unas 10 veces la prevalencia (0.063)
    assert m_xgb["auc_pr"] >= m_rl["auc_pr"] - 0.002
