"""
Preprocesamiento del CNV (MINSA) para la predicción de bajo peso al nacer.

Implementa las decisiones de la sección 10 de notebooks/01_eda.ipynb. Todos los
notebooks de modelado deben usar este módulo para que los resultados sean comparables.

Flujo de uso:
    df_raw, reporte = cargar_cnv()
    df = limpiar(df_raw)
    X, y = seleccionar_features(df, escenario="A", target="bajo_peso")
    X_train, X_test, y_train, y_test = dividir(X, y)
    prep = construir_preprocesador(escenario="A")
    X_train_t = prep.fit_transform(X_train)   # fit SOLO con train
    X_test_t = prep.transform(X_test)
"""

import io
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

URL_ZIP = "https://raw.githubusercontent.com/jasnai10/AI_Project/main/proyecto/data/raw/dataset.zip"
SEMILLA = 42
UMBRAL_BAJO_PESO = 2500  # g, criterio OMS

# Rangos válidos (EDA, sección 4.3); fuera de ellos el valor pasa a NaN
RANGOS = {
    "PESO_NACIDO": (500, 6000),
    "DUR_EMB_PARTO": (20, 45),
    "Edad_Madre": (10, 55),
}

DEPARTAMENTOS = {
    "01": "Amazonas", "02": "Áncash", "03": "Apurímac", "04": "Arequipa", "05": "Ayacucho",
    "06": "Cajamarca", "07": "Callao", "08": "Cusco", "09": "Huancavelica", "10": "Huánuco",
    "11": "Ica", "12": "Junín", "13": "La Libertad", "14": "Lambayeque", "15": "Lima",
    "16": "Loreto", "17": "Madre de Dios", "18": "Moquegua", "19": "Pasco", "20": "Piura",
    "21": "Puno", "22": "San Martín", "23": "Tacna", "24": "Tumbes", "25": "Ucayali",
}

# Agrupación de categorías raras (EDA, sección 6)
MAPA_TIPO_PARTO = {"UNICO": "UNICO", "DOBLE": "MULTIPLE", "TRIPLE": "MULTIPLE", "MAS DE TRES": "MULTIPLE"}
MAPA_FINANCIADOR = {
    "SANIDAD EP": "SANIDAD FFAA/PNP", "SANIDAD FAP": "SANIDAD FFAA/PNP",
    "SANIDAD NAVAL": "SANIDAD FFAA/PNP", "SANIDAD PNP": "SANIDAD FFAA/PNP",
}
MAPA_NIVEL = {
    "INCIAL/PRE-ESCOLAR": "NINGUN NIVEL O INICIAL",   # corrige además la errata del dataset
    "NINGUN NIVEL/ILETRADO": "NINGUN NIVEL O INICIAL",
}

CORTES_EDAD = [9, 17, 24, 34, 39, 55]
ETIQUETAS_EDAD = ["<=17", "18-24", "25-34", "35-39", ">=40"]

# Features finales (EDA, sección 10)
CATEGORICAS = ["Nivel_Instruccion", "Financiador_Parto", "Tipo_Parto", "sexo_nacido",
               "departamento", "pais_grupo"]
NUMERICAS = ["Num_embar_madre", "nacmuer_abort_madre"]
SOLO_ESCENARIO_B = ["DUR_EMB_PARTO"]


# ---------------------------------------------------------------------------
# 1. Carga
# ---------------------------------------------------------------------------

def _obtener_zip(ruta_zip=None, url=URL_ZIP):
    """Devuelve la ruta a un ZIP válido: el local si existe y no es un puntero de Git LFS; si no, lo descarga."""
    if ruta_zip is not None:
        ruta_zip = Path(ruta_zip)
        if ruta_zip.exists():
            with open(ruta_zip, "rb") as f:
                if f.read(2) == b"PK":
                    return ruta_zip
    import requests
    destino = Path(tempfile.gettempdir()) / "dataset.zip"
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    if not r.content.startswith(b"PK"):
        raise ValueError(f"La URL no devolvió un ZIP válido: {r.content[:80]!r}")
    destino.write_bytes(r.content)
    return destino


def _reparar_csv(zip_path, out_csv, sep=";", col_libre="DESC_OCUPACION"):
    """Une los ';' sobrantes dentro de DESC_OCUPACION (388 filas en el corte 30/11/2025)."""
    with zipfile.ZipFile(zip_path) as z:
        nombre = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        with z.open(nombre) as fin, open(out_csv, "w", encoding="utf-8", newline="") as fout:
            lector = io.TextIOWrapper(fin, encoding="utf-8-sig", newline="")
            header = lector.readline().rstrip("\r\n").split(sep)
            n, idx = len(header), header.index(col_libre)
            fout.write(sep.join(header) + "\n")
            rep = {"archivo": nombre, "filas": 0, "reparadas": 0, "descartadas": 0,
                   "campos_por_fila": Counter()}
            for linea in lector:
                campos = linea.rstrip("\r\n").split(sep)
                rep["campos_por_fila"][len(campos)] += 1
                extra = len(campos) - n
                if extra > 0:
                    texto = ", ".join(c.strip() for c in campos[idx: idx + extra + 1])
                    campos = campos[:idx] + [texto] + campos[idx + extra + 1:]
                    rep["reparadas"] += 1
                elif extra < 0:
                    rep["descartadas"] += 1
                    continue
                fout.write(sep.join(campos) + "\n")
                rep["filas"] += 1
    rep["campos_por_fila"] = dict(rep["campos_por_fila"])
    return rep


def cargar_cnv(ruta_zip=None, url=URL_ZIP, csv_reparado=None, verbose=True):
    """
    Carga el CNV original con todas las columnas como 'category'.

    ruta_zip: ruta local a data/raw/dataset.zip (si no existe, se descarga desde url).
    csv_reparado: dónde escribir el CSV reparado (~830 MB). Por defecto, la carpeta
                  temporal del sistema, fuera del repositorio.
    Devuelve (df, reporte).
    """
    csv_reparado = Path(csv_reparado or Path(tempfile.gettempdir()) / "cnv_reparado.csv")
    reporte = _reparar_csv(_obtener_zip(ruta_zip, url), csv_reparado)
    df = pd.read_csv(csv_reparado, sep=";", dtype="category")
    if verbose:
        print(f"Cargado {reporte['archivo']}: {df.shape[0]:,} filas, {df.shape[1]} columnas "
              f"| filas reparadas: {reporte['reparadas']} | descartadas: {reporte['descartadas']}")
    return df, reporte


# ---------------------------------------------------------------------------
# 2. Limpieza
# ---------------------------------------------------------------------------

def _cat_a_num(s, mapa=None):
    """Convierte una columna 'category' a número operando sobre sus categorías (rápido en millones de filas)."""
    cats = pd.Series(s.cat.categories.astype(str).str.strip())
    if mapa:
        cats = cats.replace(mapa)
    vals = pd.to_numeric(cats, errors="coerce").to_numpy(dtype=float)
    codes = s.cat.codes.to_numpy()
    return pd.Series(np.where(codes >= 0, vals[codes], np.nan), index=s.index)


def _limpiar_cat(s, mapa=None, nulos=("IGNORADO",)):
    """Quita espacios, aplica un mapa de recodificación y convierte los códigos de faltante en NaN."""
    d = {}
    for c in s.cat.categories:
        v = str(c).strip()
        v = (mapa or {}).get(v, v)
        d[c] = np.nan if v in nulos else v
    return s.map(d).astype("category")


def limpiar(df_raw, verbose=True):
    """
    Aplica las reglas de limpieza del EDA (secciones 4.7 y 10) y devuelve un DataFrame
    con las features candidatas, los dos targets y el año. Conserva el índice original.

    No elimina filas por faltantes en las features: eso lo hace seleccionar_features(),
    porque el conjunto de columnas depende del escenario. Solo elimina los registros sin
    peso válido, que no tienen target.
    """
    out = pd.DataFrame(index=df_raw.index)

    # Numéricas con rango válido
    for c, (lo, hi) in RANGOS.items():
        x = _cat_a_num(df_raw[c])
        out[c] = x.where(x.between(lo, hi))

    # Categóricas
    out["Nivel_Instruccion"] = _limpiar_cat(df_raw["Nivel_Intrucción_Madre"], MAPA_NIVEL)
    out["Financiador_Parto"] = _limpiar_cat(df_raw["Financiador_Parto"], MAPA_FINANCIADOR)
    out["Tipo_Parto"] = _limpiar_cat(df_raw["Tipo_Parto"], MAPA_TIPO_PARTO)
    out["sexo_nacido"] = _limpiar_cat(df_raw["sexo_nacido"])

    # Historia obstétrica: -1 e IGNORADO son faltantes; ">=5" se trata como 5 y "11 a más" como 11
    out["Num_embar_madre"] = _cat_a_num(df_raw["Num_embar_madre"], {">=5": "5", "-1": "nan"})
    out["nacmuer_abort_madre"] = _cat_a_num(df_raw["nacmuer_abort_madre"],
                                            {"NINGUNO": "0", "IGNORADO": "nan", "11 a más": "11"})

    # Variables derivadas
    ubi = df_raw["IdUbigeoInei"].astype(str).str.strip()
    out["departamento"] = ubi.where(ubi != "-1").str.zfill(6).str[:2].map(DEPARTAMENTOS).astype("category")
    pais = df_raw["Pais_Madre"].astype(str).str.strip()
    out["pais_grupo"] = pais.where(pais.isin(["PERU", "VENEZUELA"]), "OTRO").astype("category")
    out["edad_grupo"] = pd.cut(out["Edad_Madre"], CORTES_EDAD, labels=ETIQUETAS_EDAD)

    # Año, eliminación de registros sin peso válido y target binario
    out["anio"] = df_raw["FecNac_Año"].astype(int)
    n0 = len(out)
    out = out[out["PESO_NACIDO"].notna()].copy()
    out["bajo_peso"] = (out["PESO_NACIDO"] < UMBRAL_BAJO_PESO).astype(int)
    if verbose:
        print(f"limpiar(): {n0:,} → {len(out):,} filas (eliminadas {n0 - len(out):,} sin peso válido) "
              f"| prevalencia de bajo peso: {out['bajo_peso'].mean()*100:.2f}%")
    return out


# ---------------------------------------------------------------------------
# 3. Selección de features por escenario
# ---------------------------------------------------------------------------

def columnas_features(escenario="A", edad="grupo"):
    """
    Devuelve (categoricas, numericas) para el escenario.

    escenario: "A" (prenatal) o "B" (A + DUR_EMB_PARTO).
    edad: "grupo" usa edad_grupo como categórica (captura la relación en U);
          "continua" usa Edad_Madre como numérica.
    """
    escenario = escenario.upper()
    if escenario not in ("A", "B"):
        raise ValueError("escenario debe ser 'A' o 'B'")
    if edad not in ("grupo", "continua"):
        raise ValueError("edad debe ser 'grupo' o 'continua'")
    cat = CATEGORICAS + (["edad_grupo"] if edad == "grupo" else [])
    num = NUMERICAS + (["Edad_Madre"] if edad == "continua" else [])
    if escenario == "B":
        num = num + SOLO_ESCENARIO_B
    return cat, num


def seleccionar_features(df, escenario="A", target="bajo_peso", edad="grupo", verbose=True):
    """
    Devuelve (X, y) con eliminación de casos completos sobre las features del escenario.

    target: "bajo_peso" (clasificación) o "PESO_NACIDO" (regresión).
    """
    if target not in ("bajo_peso", "PESO_NACIDO"):
        raise ValueError("target debe ser 'bajo_peso' o 'PESO_NACIDO'")
    cat, num = columnas_features(escenario, edad)
    cols = cat + num
    completos = df[cols].notna().all(axis=1)
    X = df.loc[completos, cols].copy()
    for c in cat:
        X[c] = X[c].cat.remove_unused_categories()   # se mantiene 'category': ahorra memoria
    y = df.loc[completos, target].copy()
    if verbose:
        print(f"Escenario {escenario.upper()} | target={target} | {len(cols)} features | "
              f"casos completos: {completos.sum():,} de {len(df):,} ({completos.mean()*100:.1f}%)")
    return X, y


# ---------------------------------------------------------------------------
# 4. Partición y preprocesador
# ---------------------------------------------------------------------------

def dividir(X, y, test_size=0.2, semilla=SEMILLA):
    """
    Split train/test. Estratifica por y cuando el target es binario.
    Usar siempre la misma semilla para que todos los modelos se evalúen sobre el mismo test.
    """
    estratificar = y if y.nunique() == 2 else None
    return train_test_split(X, y, test_size=test_size, random_state=semilla, stratify=estratificar)


def construir_preprocesador(escenario="A", edad="grupo"):
    """
    ColumnTransformer sin ajustar: one-hot para categóricas y estandarización para numéricas.
    Debe ajustarse (fit) SOLO con los datos de entrenamiento.

    La salida es una matriz dispersa (sparse) cuando la mayoría de columnas son one-hot,
    lo que ahorra memoria con millones de filas. Si un modelo exige matriz densa,
    convertir con .toarray() sobre una submuestra.
    """
    cat, num = columnas_features(escenario, edad)
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", StandardScaler(), num),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
