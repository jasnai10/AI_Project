"""
Configuración común de las pruebas.

- Agrega proyecto/ al path para importar src.preprocessing, igual que los notebooks.
- Opción --lento: ejecuta también las pruebas marcadas como lentas (usan el CNV real de
  data/raw/dataset.zip y tardan unos minutos). Sin la opción, se saltan.
- Fixtures con el CNV sintético ya procesado con src/preprocessing.py y con las funciones
  modelo_xgboost() y evaluar() tomadas del propio notebook 06_xgboost.ipynb.
"""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PROYECTO = Path(__file__).resolve().parents[1]
NOTEBOOK = PROYECTO / "notebooks" / "06_xgboost.ipynb"
sys.path.insert(0, str(PROYECTO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessing import (SEMILLA, construir_preprocesador, dividir_temporal, limpiar,  # noqa: E402
                               seleccionar_features)

from datos_sinteticos import cnv_crudo_sintetico  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--lento", action="store_true", default=False,
                     help="ejecuta también las pruebas lentas con el CNV real (data/raw/dataset.zip)")


def pytest_configure(config):
    config.addinivalue_line("markers", "lento: prueba lenta con el CNV real; se ejecuta solo con --lento")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--lento"):
        return
    saltar = pytest.mark.skip(reason="prueba lenta con el CNV real: usar --lento para ejecutarla")
    for item in items:
        if "lento" in item.keywords:
            item.add_marker(saltar)


def preparar(df_crudo):
    """Aplica el mismo flujo que los notebooks: limpiar, 9 features, partición temporal y preprocesador."""
    df = limpiar(df_crudo, verbose=False)
    X, y = seleccionar_features(df, target="bajo_peso", verbose=False)
    X_tr, X_val, X_te, y_tr, y_val, y_te = dividir_temporal(X, y, df["anio"])
    prep = construir_preprocesador().fit(X_tr)          # fit SOLO con train
    return SimpleNamespace(X_tr=X_tr, X_val=X_val, X_te=X_te, y_tr=y_tr, y_val=y_val, y_te=y_te, prep=prep,
                           Xtr_t=prep.transform(X_tr), Xval_t=prep.transform(X_val), Xte_t=prep.transform(X_te),
                           ratio=float((y_tr == 0).sum() / y_tr.sum()))


@pytest.fixture(scope="session")
def datos():
    """CNV sintético (relaciones conocidas, sin interacciones) ya preparado con src/preprocessing.py."""
    return preparar(cnv_crudo_sintetico(n=20_000, semilla=0))


@pytest.fixture(scope="session")
def datos_con_interaccion():
    """CNV sintético con un efecto que solo aparece al combinar dos variables (o exclusivo)."""
    return preparar(cnv_crudo_sintetico(n=30_000, semilla=1, interaccion=700))


def funciones_del_notebook(nombres=("modelo_xgboost", "evaluar")):
    """
    Toma del notebook las funciones indicadas (y las constantes simples, como UMBRAL o MAX_ARBOLES)
    sin ejecutar el resto de sus celdas. Así las pruebas usan exactamente el modelo y las métricas
    definidos en el notebook, no una copia que podría quedar desactualizada.
    """
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    from xgboost import XGBClassifier

    espacio = {"np": np, "pd": pd, "XGBClassifier": XGBClassifier, "SEMILLA": SEMILLA,
               "accuracy_score": accuracy_score, "precision_score": precision_score, "recall_score": recall_score,
               "f1_score": f1_score, "roc_auc_score": roc_auc_score,
               "average_precision_score": average_precision_score}
    celdas = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    for celda in celdas:
        if celda["cell_type"] != "code":
            continue
        fuente = "".join(celda["source"])
        try:
            arbol = ast.parse(fuente)
        except SyntaxError:      # celdas con comandos de IPython (!pip, !git): no definen el modelo
            continue
        for nodo in arbol.body:
            es_constante = isinstance(nodo, ast.Assign) and isinstance(nodo.value, ast.Constant)
            if es_constante or (isinstance(nodo, ast.FunctionDef) and nodo.name in nombres):
                exec(compile(ast.Module(body=[nodo], type_ignores=[]), str(NOTEBOOK), "exec"), espacio)
    faltan = [n for n in nombres if n not in espacio]
    if faltan:
        raise LookupError(f"El notebook {NOTEBOOK.name} no define {faltan}")
    return SimpleNamespace(**{k: espacio[k] for k in (*nombres, "UMBRAL", "MAX_ARBOLES") if k in espacio})


@pytest.fixture(scope="session")
def nb():
    return funciones_del_notebook()
