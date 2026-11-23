"""
CNV sintético para las pruebas: imita el formato crudo que devuelve cargar_cnv().

Genera solo las 11 columnas que usa limpiar(), todas como 'category' de texto y con los
mismos códigos que el CNV real (por ejemplo "IGNORADO" y "-1" como faltantes, ">=5" en
Num_embar_madre o " PERU " con espacios en Pais_Madre), así que pasa por limpiar(),
seleccionar_features() y dividir_temporal() igual que los datos reales.

El peso al nacer se simula con relaciones conocidas: depende sobre todo de la duración del
embarazo (como en el CNV, donde es la variable dominante) y en menor medida del tipo de parto,
el sexo, la edad de la madre y el nivel de instrucción. Con interaccion > 0 se agrega además un
efecto que solo aparece al combinar dos variables (sexo del recién nacido y financiador del
parto, en forma de "o exclusivo"): una regresión logística, que suma efectos independientes, no
puede representarlo, pero un modelo de árboles como XGBoost sí.
"""

import numpy as np
import pandas as pd

ANIOS = list(range(2015, 2026))

# Distribuciones aproximadas del CNV real (corte 30/11/2025)
SEMANAS = {20: 0.0003, 22: 0.0002, 24: 0.0005, 26: 0.0007, 28: 0.0011, 30: 0.0018, 31: 0.0019, 32: 0.0035,
           33: 0.0046, 34: 0.0087, 35: 0.0137, 36: 0.0297, 37: 0.0860, 38: 0.2218, 39: 0.3290, 40: 0.2480,
           41: 0.0427, 42: 0.0038}
NIVEL = {"SECUNDARIA COMPLETA": 0.351, "SECUNDARIA INCOMPLETA": 0.168, "SUPERIOR NO UNIV. COMPLETA": 0.116,
         "SUPERIOR UNIV. COMPLETA": 0.115, "PRIMARIA COMPLETA": 0.080, "SUPERIOR NO UNIV. INCOMPLETA": 0.059,
         "PRIMARIA INCOMPLETA": 0.057, "SUPERIOR UNIV. INCOMPLETA": 0.046, "NINGUN NIVEL/ILETRADO": 0.006,
         "INCIAL/PRE-ESCOLAR": 0.001, "IGNORADO": 0.001}
FINANCIADOR = {"SIS": 0.700, "ESSALUD": 0.180, "PARTICULAR": 0.058, "PRIVADOS": 0.049, "IGNORADO": 0.008,
               "SANIDAD PNP": 0.003, "SANIDAD NAVAL": 0.001, "SANIDAD EP": 0.0005, "SANIDAD FAP": 0.0005}
TIPO_PARTO = {"UNICO": 0.955, "DOBLE": 0.035, "TRIPLE": 0.005, "MAS DE TRES": 0.001, "IGNORADO": 0.004}
SEXO = {"MASCULINO": 0.510, "FEMENINO": 0.489, "IGNORADO": 0.001}
EMBARAZOS = {"1": 0.300, "2": 0.288, "3": 0.198, "4": 0.107, ">=5": 0.097, "-1": 0.010}
PAIS = {" PERU ": 0.970, " VENEZUELA ": 0.025, " COLOMBIA ": 0.003, " BOLIVIA ": 0.002}


def _elegir(rng, dist, n):
    valores = list(dist)
    p = np.array(list(dist.values()), dtype=float)
    return rng.choice(valores, size=n, p=p / p.sum())


def cnv_crudo_sintetico(n=20_000, semilla=0, interaccion=0.0):
    """
    DataFrame con el formato crudo del CNV (columnas 'category' de texto), listo para limpiar().

    n: número de registros (repartidos al azar entre 2015 y 2025).
    semilla: semilla del generador; la misma semilla da exactamente los mismos datos.
    interaccion: gramos que se restan al peso cuando el recién nacido es de sexo femenino
                 o el parto lo financia el SIS, pero no ambos (efecto de "o exclusivo").
    """
    rng = np.random.default_rng(semilla)
    anio = rng.choice(ANIOS, size=n)
    semanas = _elegir(rng, SEMANAS, n).astype(int)
    tipo = _elegir(rng, TIPO_PARTO, n)
    sexo = _elegir(rng, SEXO, n)
    nivel = _elegir(rng, NIVEL, n)
    financiador = _elegir(rng, FINANCIADOR, n)
    embarazos = _elegir(rng, EMBARAZOS, n)
    pais = _elegir(rng, PAIS, n)
    edad = np.clip(np.round(rng.normal(28, 6.5, n)), 12, 50).astype(int)
    depto = rng.choice([f"{d:02d}" for d in range(1, 26)], size=n)
    ubigeo = np.char.add(depto, "0101").astype(object)

    # Peso al nacer (g): la duración del embarazo domina; el resto ajusta poco
    peso = 3250 + 180 * (semanas - 39)
    peso = peso - 550 * np.isin(tipo, ["DOBLE", "TRIPLE", "MAS DE TRES"])
    peso = peso - 110 * (sexo == "FEMENINO")
    peso = peso - 90 * (edad <= 17) - 60 * (edad >= 40)
    peso = peso - 70 * np.isin(nivel, ["PRIMARIA INCOMPLETA", "NINGUN NIVEL/ILETRADO", "INCIAL/PRE-ESCOLAR"])
    if interaccion:
        peso = peso - interaccion * ((sexo == "FEMENINO") ^ (financiador == "SIS"))
    peso = np.clip(np.round(peso + rng.normal(0, 300, n)), 500, 6000).astype(int)

    peso_txt = np.char.zfill(peso.astype(str), 4).astype(object)
    semanas_txt = semanas.astype(str).astype(object)
    edad_txt = edad.astype(str).astype(object)

    # Algunos códigos de faltante y valores fuera de rango, como en el CNV real
    def ensuciar(columna, codigo, proporcion):
        idx = rng.random(n) < proporcion
        columna[idx] = codigo

    ensuciar(peso_txt, "-1", 0.002)          # sin peso válido -> limpiar() elimina la fila
    ensuciar(semanas_txt, "99", 0.002)       # fuera de rango -> NaN -> fuera por casos completos
    ensuciar(ubigeo, "-1", 0.002)            # departamento desconocido -> NaN

    df = pd.DataFrame({
        "FecNac_Año": anio.astype(str),
        "PESO_NACIDO": peso_txt,
        "DUR_EMB_PARTO": semanas_txt,
        "sexo_nacido": sexo,
        "Tipo_Parto": tipo,
        "Edad_Madre": edad_txt,
        "Nivel_Intrucción_Madre": nivel,
        "Num_embar_madre": embarazos,
        "Pais_Madre": pais,
        "IdUbigeoInei": ubigeo,
        "Financiador_Parto": financiador,
    })
    return df.astype("category")
