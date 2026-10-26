# Clasificador de PQRS – Cartago, Valle del Cauca

Proyecto de Inteligencia Artificial. Analiza los tiempos de respuesta de las
Peticiones, Quejas, Reclamos y Sugerencias (PQRS) del municipio de Cartago y
prepara los datos para un modelo que priorice los trámites con mayor
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
│   ├── preparacion_pqrs.py      # Clase 7 y Avance 2: Pandas + Seaborn + preparación para ML
│   ├── neurona_pqrs.py          # Clase 8: neurona artificial para riesgo de demora
│   └── perceptron_proyecto.py   # Clase 9 y Avance 2: perceptrón desde cero con NumPy
├── data/
│   ├── pqrs_cartago.csv                 # Dataset original (12 registros)
│   ├── proyecto_limpio.csv              # Dataset limpio con las variables del modelo
│   ├── pqrs_preparadas_ml.csv           # Dataset codificado con get_dummies
│   ├── informe_proyecto.md              # Generado por analisis_proyecto.py
│   ├── eda_*.png                        # Generados por eda_proyecto.py
│   ├── sns_*.png                        # Generados por preparacion_pqrs.py (5 gráficos)
│   ├── neurona_pqrs_error.png           # Generado por neurona_pqrs.py
│   ├── curva_aprendizaje_proyecto.png   # Generado por perceptron_proyecto.py
│   └── frontera_decision_proyecto.png   # Generado por perceptron_proyecto.py
├── notebooks/
└── tests/
```

## Cómo ejecutar el proyecto

### Con Docker

```bash
cd ~/ia-python

# 1. Levantar el contenedor (--build reconstruye la imagen con el código y las librerías)
docker compose up -d --build

# 2. Entrar al contenedor y a la carpeta del código
docker exec -it ia_python bash
cd /app/src

# 3. Ejecutar los scripts (en este orden)
python analisis_proyecto.py
python eda_proyecto.py
python preparacion_pqrs.py
python neurona_pqrs.py
python perceptron_proyecto.py

# 4. Salir y apagar
exit
docker compose down
```

### Sin Docker (WSL)

```bash
cd ~/ia-python
python3 src/preparacion_pqrs.py
python3 src/perceptron_proyecto.py
```

Requiere `numpy`, `pandas`, `matplotlib` y `seaborn` (ver `requirements.txt`).

## Dataset

`data/pqrs_cartago.csv` tiene 12 registros y 4 columnas. Cada fila es un conteo
agregado de radicados por tipo de trámite.

| Columna | Descripción |
|---|---|
| `tipo` | Petición, Queja, Reclamo o Sugerencia |
| `radicados` | Cantidad de radicados |
| `tiempo_respuesta_dias` | Tiempo de respuesta en días |
| `dependencia` | Planeación, Obras Públicas o Hacienda |

## Preparación de Datos (Clase 7)

Script: `src/preparacion_pqrs.py` (Pandas + Seaborn).

- **Nulos y duplicados:** el dataset no tiene nulos ni duplicados. El script
  incluye la lógica de limpieza (mediana para numéricas, moda para categóricas).
- **Codificación:** One-Hot (`pd.get_dummies`) para `tipo` y `dependencia`
  (nominales) y Label Encoding para `riesgo_demora` (ordinal).
- **Columna derivada `riesgo_demora`:** bajo (hasta 5 días), medio (6 a 8) y
  alto (más de 8 días): 5 registros bajos, 3 medios y 4 altos.

### Hallazgos

1. **El tiempo de respuesta depende del tipo de trámite.** Promedios:
   Sugerencia 3.33 días, Petición 5.33, Queja 8.00 y Reclamo 10.33. El
   promedio general es 6.75 días (mínimo 3, máximo 11).
2. **El volumen de radicados no explica la demora.** La correlación lineal
   entre `radicados` y `tiempo_respuesta_dias` es débil (0.12).
3. **Por dependencia:** Hacienda es la más lenta (10.33 días), Obras Públicas
   está en el medio (8.00) y Planeación es la más rápida (4.33).
4. **`tipo` y `dependencia` están casi superpuestas:** todos los Reclamos son de
   Hacienda, todas las Quejas de Obras Públicas y las Peticiones y Sugerencias
   de Planeación.
5. **Fuga de información:** `riesgo_demora` se calcula con
   `tiempo_respuesta_dias`, por lo que un clasificador no debe usar el tiempo
   como entrada.

## Clase 8: neurona para predecir riesgo de demora

Script: `src/neurona_pqrs.py` (Python puro). Una neurona con función escalón
aprende a predecir si un trámite tiene riesgo de demora alto (más de 8 días).
Entradas: radicados (normalizado), `es_queja` y `es_reclamo`.

- Llegó a error 0 en 107 épocas y acertó 12 de 12 trámites.
- Limitación: se evaluó con los mismos 12 datos con los que se entrenó.

## Avance 2

### Qué se hizo

1. **Preparación de datos (`src/preparacion_pqrs.py`):** inspección con `shape`,
   `info()`, nulos y duplicados; diagnóstico de atípicos, valores imposibles y
   categorías inconsistentes; tabla antes/después; exportación a
   `data/proyecto_limpio.csv`.
2. **Visualización:** 5 gráficos con Seaborn (`data/sns_*.png`); el gráfico
   `sns_entradas_vs_objetivo.png` relaciona las entradas del modelo con el objetivo.
3. **Perceptrón desde cero (`src/perceptron_proyecto.py`):** sigmoide, pérdida
   logarítmica y descenso del gradiente con NumPy, sin Scikit-Learn.

### Tabla antes / después

| Medida | Antes | Después |
|---|---|---|
| Filas | 12 | 12 |
| Columnas | 4 | 7 (3 variables nuevas) |
| Nulos | 0 | 0 |
| Duplicados | 0 | 0 |

### Variables del modelo

- **Objetivo `demora_alta`:** 1 si `tiempo_respuesta_dias` supera la mediana
  (6.5 días), 0 en caso contrario. Resultan 6 registros de cada clase.
  `tiempo_respuesta_dias` no se usa como entrada.
- **Entradas (estandarizadas):** `radicados` y `carga_dependencia` (total de
  radicados de la dependencia: Planeación 192, Obras Públicas 97, Hacienda 79).

### Resultados del perceptrón

| Resultado | Valor |
|---|---|
| Pérdida inicial | 0.6931 |
| Pérdida final | 0.0232 |
| Exactitud final | 100 % |
| w1 (radicados) | -0.0633 |
| w2 (carga_dependencia) | -3.8841 |
| b | 0.0616 |

- `w2` negativo: a mayor carga de la dependencia, menor probabilidad de demora
  alta; Hacienda y Obras Públicas (menos radicados en total) son las más lentas.
- `w1` casi cero: el número de radicados de un trámite no influye en la decisión.
- Los datos son linealmente separables: la pérdida sigue bajando y no hay errores.

### Limitaciones

- La exactitud se mide con los mismos 12 registros del entrenamiento (sin datos
  de prueba), por lo que no demuestra que el modelo funcione con trámites nuevos.
- `carga_dependencia` solo toma 3 valores, así que el modelo aprendió qué
  dependencias son lentas.
- Con el umbral de 8 días en lugar de la mediana, la exactitud baja a 83.33 %.
- El dataset es pequeño y agregado; para el proyecto final se ampliará.

## Próximos pasos

- [x] Cargar y limpiar datos de PQRS
- [x] EDA con NumPy + Matplotlib
- [x] Preparación de datos con Pandas y Seaborn
- [x] Neurona artificial en Python puro para riesgo de demora (Clase 8)
- [x] Perceptrón desde cero con NumPy para demora alta (Avance 2)
- [ ] Ampliar el dataset y separar datos de entrenamiento y prueba
- [ ] Comparar modelos de Scikit-Learn (Clase 12)
- [ ] Clasificador de prioridad (bajo / medio / alto riesgo)
- [ ] Dashboard en tiempo real para funcionarios
- [ ] Chatbot ciudadano para consultar el estado de un trámite