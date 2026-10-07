# Informe del Proyecto: Clasificador de PQRS - Cartago

## Descripcion del dataset

El dataset contiene registros de Peticiones, Quejas, Reclamos y Sugerencias (PQRS) atendidas por distintas dependencias del municipio de Cartago, Valle del Cauca. Cada fila representa un conteo agregado de radicados por tipo de tramite, junto con su tiempo de respuesta en dias y la dependencia responsable.

## Estadisticas calculadas

| Estadistica | Valor |
|---|---|
| Total de registros | 12 |
| Total de radicados | 368 |
| Tiempo de respuesta promedio general | 6.75 dias |
| Tiempo de respuesta maximo | 11 dias |
| Tiempo de respuesta minimo | 3 dias |
| Tipo de tramite con mayor volumen | Petición |
| Dependencia mas lenta (promedio) | Hacienda (10.33 dias) |
| Dependencia mas rapida (promedio) | Planeación (4.33 dias) |

### Tiempo de respuesta promedio por tipo de tramite

| Tipo de tramite | Tiempo promedio (dias) |
|---|---|
| Sugerencia | 3.33 |
| Petición | 5.33 |
| Queja | 8.00 |
| Reclamo | 10.33 |

## Interpretacion de resultados

El tiempo de respuesta promedio general es de 6.75 dias, pero varia significativamente segun el tipo de tramite y la dependencia. La dependencia **Hacienda** es la que mas tarda en responder, mientras que **Planeación** es la mas eficiente. El tipo de tramite con mayor volumen de radicados es **Petición**.

Para el proyecto de Clasificador de PQRS, esto sugiere que un modelo de clasificacion o priorizacion deberia tomar en cuenta tanto el tipo de tramite como la dependencia asignada, ya que ambos factores estan relacionados con el tiempo esperado de resolucion. Esta informacion puede usarse mas adelante para entrenar un modelo que prediga el tiempo de respuesta o priorice automaticamente los tramites con mayor riesgo de demora.
