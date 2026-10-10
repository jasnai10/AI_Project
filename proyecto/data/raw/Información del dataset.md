# Descripción del dataset

## Origen

El conjunto de datos **Registros de Nacidos Vivos en el Perú (2015–2025)** proviene del Certificado de Nacido Vivo (CNV), documento administrado por el Ministerio de Salud del Perú que se emite con ocasión de cada nacimiento registrado en el territorio nacional. El dataset se encuentra publicado en la Plataforma Nacional de Datos Abiertos del Gobierno del Perú, bajo la condición de datos públicos de libre acceso.

**Enlace de acceso:** https://www.datosabiertos.gob.pe/dataset/registros-de-nacidos-vivos-en-el-perú-2015–2025

## Características generales

| Atributo | Descripción |
|---|---|
| Entidad responsable | Ministerio de Salud (MINSA) |
| Número de registros | 4 782 338 |
| Periodo cubierto | 2015 – 2025 (corte al 30 de noviembre de 2025) |
| Número de variables | 22 |
| Cobertura geográfica | Nacional, con desagregación por ubigeo distrital |
| Formato | CSV |

El diseño del conjunto, con variables normalizadas y definiciones explícitas, asegura consistencia semántica y operativa, lo que lo convierte en un recurso adecuado para estudios de salud pública, epidemiología perinatal y evaluación de políticas sanitarias.

## Estructura de variables

A continuación se presentan las variables del dataset, agrupadas en cinco bloques temáticos.

### Características del recién nacido

| Variable | Descripción | Tipo |
|---|---|---|
| `PESO_NACIDO` | Peso del recién nacido al momento del nacimiento, expresado en gramos | INT |
| `TALLA_NACIDO` | Talla del recién nacido al momento del nacimiento, expresada en centímetros con hasta un decimal | INT |
| `sexo_nacido` | Sexo del recién nacido: masculino o femenino | VARCHAR |
| `FecNac_Año` | Año de la fecha de nacimiento, en formato de cuatro dígitos | VARCHAR |
| `FecNac_Mes` | Mes de la fecha de nacimiento, en formato de dos dígitos | VARCHAR |

### Características de la madre

| Variable | Descripción | Tipo |
|---|---|---|
| `Edad_Madre` | Edad de la madre al momento del parto, expresada en años | INT |
| `Estado_Civil` | Estado civil de la madre al momento del parto: casada, conviviente, divorciada, separada, soltera, viuda o ignorado | VARCHAR |
| `Nivel_Intrucción_Madre` | Nivel de instrucción de la madre al momento del parto: ignorado, inicial/pre-escolar, ningún nivel/iletrado, primaria incompleta, primaria completa, secundaria incompleta, secundaria completa, superior no universitaria incompleta, superior no universitaria completa, superior universitaria incompleta o superior universitaria completa | VARCHAR |
| `DESC_OCUPACION` | Ocupación de la madre al momento del parto, declarada por la propia madre | VARCHAR |
| `Pais_Madre` | País de origen de la madre al momento del parto, según la lista oficial de países | VARCHAR |
| `IdUbigeoInei` | Código de ubigeo del lugar de domicilio de la madre al momento del parto, según el listado oficial del INEI, en formato de seis dígitos; -1 en caso de ignorado | VARCHAR |

### Historia obstétrica

| Variable | Descripción | Tipo |
|---|---|---|
| `Num_embar_madre` | Número de embarazos previos de la madre al momento del parto: -1 (ignorado), 1, 2, 3, 4, >=5 | VARCHAR |
| `Hijos_vivo_madre` | Número de hijos vivos de la madre al momento del parto: -1 (ignorado), 1, 2, 3, 4, >=5 | VARCHAR |
| `Hijos_fallec_madre` | Número de hijos fallecidos de la madre al momento del parto: -1 (ignorado), 1, 2, 3, 4, >=5 | VARCHAR |
| `nacmuer_abort_madre` | Número de abortos o nonatos de la madre al momento del parto: ninguno, ignorado, 1 a 10, 11 a más | VARCHAR |

### Condiciones del embarazo y del parto

| Variable | Descripción | Tipo |
|---|---|---|
| `DUR_EMB_PARTO` | Duración del embarazo al momento del parto, expresada en semanas | INT |
| `Tipo_Parto` | Tipo de parto según la cantidad de recién nacidos: único, doble, triple, más de tres o ignorado | VARCHAR |
| `Condicion_Parto` | Condición en la que se realizó el parto: eutócico, instrumentado, cesárea o ignorado | VARCHAR |

### Atención y registro del parto

| Variable | Descripción | Tipo |
|---|---|---|
| `Lugar_Nacido` | Lugar donde se realizó el nacimiento: centro de trabajo, domicilio, establecimiento de salud, ignorado, otro o vía pública | VARCHAR |
| `Atiende_Parto` | Personal de salud o persona que atendió el parto: enfermera, familiar, ignorado, interno, médico, médico gineco-obstetra, nadie (autoayuda), obstetra, otro, otro profesional de la salud, partera, promotor de salud o técnico de salud | VARCHAR |
| `Financiador_Parto` | Método de financiamiento para la atención del parto: EsSalud, ignorado, particular, privados, sanidad EP, sanidad FAP, sanidad naval, sanidad PNP o SIS | VARCHAR |
| `Ipress` | Código único del establecimiento de salud donde se registró el certificado de nacido vivo, según el RENIPRESS; -1 en caso de ignorado | INT |

Información obtenida del diccionario de datos del MINSA.

## Variables objetivo del proyecto

El proyecto aborda el problema mediante modelos de regresión y de clasificación, por lo que se definen dos variables objetivo a partir de `PESO_NACIDO`.

| Enfoque | Variable objetivo | Definición |
|---|---|---|
| Regresión | `PESO_NACIDO` | Peso del recién nacido expresado en gramos, tratado como variable continua |
| Clasificación | `bajo_peso` | Variable binaria derivada de `PESO_NACIDO` según el umbral de bajo peso al nacer establecido por la Organización Mundial de la Salud |

```
bajo_peso = 1   si PESO_NACIDO < 2500
bajo_peso = 0   si PESO_NACIDO >= 2500
```
