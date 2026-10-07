# Clasificador de PQRS – Cartago, Valle del Cauca

Proyecto de Inteligencia Artificial. Analiza los tiempos de respuesta de las
Peticiones, Quejas, Reclamos y Sugerencias (PQRS) del municipio de Cartago y
prepara los datos para un futuro modelo que priorice los trámites con mayor
riesgo de demora.

## Estructura del proyecto

```
ia-python/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── src/
│   ├── analisis_proyecto.py     # Clase 3: lee el CSV (csv.DictReader) y genera informe_proyecto.md
│   ├── eda_proyecto.py          # Clase 4: EDA con NumPy + Matplotlib
│   ├── preparacion_pqrs.py      # Clase 7: Pandas + Seaborn + preparación para ML
│   └── neurona_pqrs.py          # Clase 8: neurona artificial para riesgo de demora
├── data/
│   ├── pqrs_cartago.csv         # Dataset original (12 registros)
│   ├── informe_proyecto.md      # Generado por analisis_proyecto.py
│   ├── eda_*.png                # Generados por eda_proyecto.py
│   ├── sns_*.png                # Generados por preparacion_pqrs.py
│   ├── pqrs_preparadas_ml.csv   # Dataset listo para ML
│   └── neurona_pqrs_error.png   # Generado por neurona_pqrs.py
├── notebooks/
└── tests/
```

## Cómo ejecutar el proyecto

```bash
cd ~/ia-python

# 1. Levantar el contenedor
docker compose up -d

# 2. Entrar al contenedor
docker exec -it ia_python bash

# 3. Ejecutar los scripts (dentro del contenedor, en /app)
python src/analisis_proyecto.py
python src/eda_proyecto.py
python src/preparacion_pqrs.py
python src/neurona_pqrs.py

# 4. Salir y apagar
exit
docker compose down
```

Si se modifica `requirements.txt` (por ejemplo, al agregar `pandas` y `seaborn`),
hay que reconstruir la imagen:

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

## Dataset

`data/pqrs_cartago.csv` tiene 12 registros y 4 columnas. Cada fila es un conteo
agregado de radicados por tipo de trámite.

| Columna | Descripción |
|---|---|
| `tipo` | Petición, Queja, Reclamo o Sugerencia |
| `radicados` | Cantidad de radicados |
| `tiempo_respuesta_dias` | Tiempo de respuesta en días |
| `dependencia` | Planeación, Obras Públicas o Hacienda |

## Preparación de Datos

Script: `src/preparacion_pqrs.py` (Pandas + Seaborn).

### Valores nulos

El dataset no tiene valores nulos (`df.isnull().sum()` da 0 en todas las
columnas) ni duplicados. Aun así, el script incluye la lógica de limpieza:
imputación con la **mediana** para columnas numéricas y con la **moda** para
categóricas. Se usa la mediana porque no se ve afectada por valores extremos,
a diferencia de la media.

### Variables categóricas codificadas

- **One-Hot Encoding** (`pd.get_dummies`) para `tipo` y `dependencia`, porque
  son variables nominales (no tienen un orden natural).
- **Label Encoding** para `riesgo_demora` (bajo = 0, medio = 1, alto = 2),
  porque es una variable ordinal.

### Columna derivada: `riesgo_demora`

Clasifica cada registro según su tiempo de respuesta: **bajo** (hasta 5 días),
**medio** (6 a 8 días) y **alto** (más de 8 días). Resultado: 5 registros de
riesgo bajo, 3 de riesgo medio y 4 de riesgo alto. Se creó porque es la base del
clasificador de prioridad previsto para el proyecto.

### Hallazgos

1. **El tiempo de respuesta depende del tipo de trámite.** Promedios:
   Sugerencia 3.33 días, Petición 5.33, Queja 8.00 y Reclamo 10.33. El
   tiempo promedio general es 6.75 días (mínimo 3, máximo 11).
2. **Los trámites con menos radicados son los más lentos.** Las Peticiones
   tienen el mayor volumen (47.67 radicados en promedio) y los Reclamos solo
   26.33, pero tardan el doble en responderse. Aun así, la correlación lineal
   entre `radicados` y `tiempo_respuesta_dias` es débil (0.12), por lo que el
   volumen por sí solo no explica la demora.
3. **Por dependencia:** Hacienda es la más lenta (10.33 días en promedio),
   Obras Públicas está en el medio (8.00) y Planeación es la más rápida (4.33).
4. **`tipo` y `dependencia` están casi superpuestas.** Todos los Reclamos son de
   Hacienda, todas las Quejas son de Obras Públicas y las Peticiones y
   Sugerencias son de Planeación. Por eso, para un modelo, la dependencia aporta
   poca información adicional frente al tipo de trámite.
5. **Cuidado con la fuga de información (data leakage).** Como `riesgo_demora`
   se calcula a partir de `tiempo_respuesta_dias`, un clasificador de riesgo no
   debe usar el tiempo de respuesta como variable de entrada.

### Visualizaciones (en `data/`)

- `sns_histograma_tiempo.png`: distribución del tiempo de respuesta.
- `sns_barras_tipo.png`: tiempo promedio por tipo de trámite.
- `sns_boxplot_dependencia.png`: tiempo de respuesta por dependencia.
- `sns_dispersion_radicados.png`: radicados vs tiempo, con línea de tendencia.

## Clase 8 aplicada: neurona para predecir riesgo de demora

Script: `src/neurona_pqrs.py` (Python puro). Una neurona artificial con función
escalón aprende a predecir si un trámite tendrá **riesgo de demora alto**
(más de 8 días de respuesta).

### Entradas y salida

| Variable | Descripción |
|---|---|
| `x1` | Radicados, normalizado entre 0 y 1 |
| `x2` | 1 si el tipo es Queja, 0 si no |
| `x3` | 1 si el tipo es Reclamo, 0 si no |
| `y` | 1 = riesgo alto, 0 = no es riesgo alto |

No se usa `tiempo_respuesta_dias` como entrada, porque con él se calcula `y`
(sería fuga de información). Tampoco se usa `dependencia`, porque repite la
información del tipo (Hacienda = Reclamos, Obras Públicas = Quejas).

### Resultados

- Parámetros iniciales en 0 y tasa de aprendizaje 0.1.
- La neurona llegó a error 0 en **107 épocas** y acertó **12 de 12** trámites.
- Parámetros aprendidos: w(radicados) = 0.351, w(es_queja) = 0.300,
  w(es_reclamo) = 0.500 y b = -0.500.
- Los Reclamos son el factor que más empuja hacia riesgo alto, seguidos de las
  Quejas. Con el bias en -0.5, un Reclamo ya activa la neurona por sí solo.

### Limitaciones

- Con solo 12 registros, el 12/12 se midió sobre los mismos datos con los que se
  entrenó. No demuestra que el modelo funcione con trámites nuevos. Falta separar
  datos de entrenamiento y de prueba.
- El peso de `radicados` depende de un solo caso: la Queja de 35 radicados (9
  días) frente a las Quejas de 30 y 32 radicados (8 y 7 días).
- Necesitó muchas más épocas que las compuertas lógicas (107 contra 2 a 4)
  porque los datos son menos "limpios" y hay más variables.

## Próximos pasos

- [x] Cargar y limpiar datos de PQRS
- [x] EDA con NumPy + Matplotlib
- [x] Preparación de datos con Pandas y Seaborn
- [x] Neurona artificial en Python puro para riesgo de demora (Clase 8)
- [ ] Modelo de regresión: predecir tiempo de respuesta
- [ ] Clasificador de prioridad (bajo / medio / alto riesgo)
- [ ] Dashboard en tiempo real para funcionarios
- [ ] Chatbot ciudadano para consultar el estado de un trámite