"""
Proyecto: Clasificador de PQRS - Cartago, Valle del Cauca
Actividad independiente - Clase 3: Manejo de archivos y estructuras de datos

Lee el dataset de PQRS (CSV), calcula estadisticas relevantes para el
proyecto y genera un informe en Markdown.
"""

import csv


def leer_datos(archivo_csv):
    """Lee el archivo CSV de PQRS y retorna una lista de diccionarios,
    con las columnas numericas ya convertidas a int."""
    registros = []
    try:
        with open(archivo_csv, "r", encoding="utf-8") as archivo:
            lector = csv.DictReader(archivo)
            for fila in lector:
                registro = {
                    "tipo": fila["tipo"],
                    "radicados": int(fila["radicados"]),
                    "tiempo_respuesta_dias": int(fila["tiempo_respuesta_dias"]),
                    "dependencia": fila["dependencia"],
                }
                registros.append(registro)
    except FileNotFoundError:
        print(f"Error: no se encontro el archivo '{archivo_csv}'. Verifique la ruta.")
        return []
    except (ValueError, KeyError) as e:
        print(f"Error al procesar una fila del archivo: {e}")
        return []

    return registros


def calcular_estadisticas(registros):
    """Calcula al menos 5 estadisticas relevantes para el proyecto de
    clasificacion/priorizacion de PQRS."""
    if not registros:
        return {}

    total_radicados = sum(r["radicados"] for r in registros)

    tiempos = [r["tiempo_respuesta_dias"] for r in registros]
    tiempo_promedio_general = sum(tiempos) / len(tiempos)

    # Promedio de tiempo de respuesta agrupado por tipo de tramite
    tiempos_por_tipo = {}
    radicados_por_tipo = {}
    for r in registros:
        tiempos_por_tipo.setdefault(r["tipo"], []).append(r["tiempo_respuesta_dias"])
        radicados_por_tipo[r["tipo"]] = radicados_por_tipo.get(r["tipo"], 0) + r["radicados"]

    promedio_por_tipo = {
        tipo: sum(valores) / len(valores) for tipo, valores in tiempos_por_tipo.items()
    }

    # Promedio de tiempo de respuesta agrupado por dependencia
    tiempos_por_dependencia = {}
    for r in registros:
        tiempos_por_dependencia.setdefault(r["dependencia"], []).append(r["tiempo_respuesta_dias"])

    promedio_por_dependencia = {
        dep: sum(valores) / len(valores) for dep, valores in tiempos_por_dependencia.items()
    }

    tipo_mayor_volumen = max(radicados_por_tipo, key=radicados_por_tipo.get)
    dependencia_mas_lenta = max(promedio_por_dependencia, key=promedio_por_dependencia.get)
    dependencia_mas_rapida = min(promedio_por_dependencia, key=promedio_por_dependencia.get)

    return {
        "total_registros": len(registros),
        "total_radicados": total_radicados,
        "tiempo_promedio_general": tiempo_promedio_general,
        "tiempo_maximo": max(tiempos),
        "tiempo_minimo": min(tiempos),
        "promedio_por_tipo": promedio_por_tipo,
        "promedio_por_dependencia": promedio_por_dependencia,
        "tipo_mayor_volumen": tipo_mayor_volumen,
        "dependencia_mas_lenta": dependencia_mas_lenta,
        "dependencia_mas_rapida": dependencia_mas_rapida,
    }


def generar_informe(estadisticas, archivo_salida):
    """Genera un informe en Markdown con las estadisticas del proyecto."""
    if not estadisticas:
        print("No hay estadisticas para generar el informe.")
        return

    with open(archivo_salida, "w", encoding="utf-8") as archivo:
        archivo.write("# Informe del Proyecto: Clasificador de PQRS - Cartago\n\n")

        archivo.write("## Descripcion del dataset\n\n")
        archivo.write(
            "El dataset contiene registros de Peticiones, Quejas, Reclamos y "
            "Sugerencias (PQRS) atendidas por distintas dependencias del "
            "municipio de Cartago, Valle del Cauca. Cada fila representa un "
            "conteo agregado de radicados por tipo de tramite, junto con su "
            "tiempo de respuesta en dias y la dependencia responsable.\n\n"
        )

        archivo.write("## Estadisticas calculadas\n\n")
        archivo.write("| Estadistica | Valor |\n")
        archivo.write("|---|---|\n")
        archivo.write(f"| Total de registros | {estadisticas['total_registros']} |\n")
        archivo.write(f"| Total de radicados | {estadisticas['total_radicados']} |\n")
        archivo.write(
            f"| Tiempo de respuesta promedio general | "
            f"{estadisticas['tiempo_promedio_general']:.2f} dias |\n"
        )
        archivo.write(f"| Tiempo de respuesta maximo | {estadisticas['tiempo_maximo']} dias |\n")
        archivo.write(f"| Tiempo de respuesta minimo | {estadisticas['tiempo_minimo']} dias |\n")
        archivo.write(
            f"| Tipo de tramite con mayor volumen | {estadisticas['tipo_mayor_volumen']} |\n"
        )
        archivo.write(
            f"| Dependencia mas lenta (promedio) | {estadisticas['dependencia_mas_lenta']} "
            f"({estadisticas['promedio_por_dependencia'][estadisticas['dependencia_mas_lenta']]:.2f} dias) |\n"
        )
        archivo.write(
            f"| Dependencia mas rapida (promedio) | {estadisticas['dependencia_mas_rapida']} "
            f"({estadisticas['promedio_por_dependencia'][estadisticas['dependencia_mas_rapida']]:.2f} dias) |\n"
        )

        archivo.write("\n### Tiempo de respuesta promedio por tipo de tramite\n\n")
        archivo.write("| Tipo de tramite | Tiempo promedio (dias) |\n")
        archivo.write("|---|---|\n")
        for tipo, promedio in sorted(estadisticas["promedio_por_tipo"].items(), key=lambda x: x[1]):
            archivo.write(f"| {tipo} | {promedio:.2f} |\n")

        archivo.write("\n## Interpretacion de resultados\n\n")
        archivo.write(
            f"El tiempo de respuesta promedio general es de "
            f"{estadisticas['tiempo_promedio_general']:.2f} dias, pero varia "
            f"significativamente segun el tipo de tramite y la dependencia. "
            f"La dependencia **{estadisticas['dependencia_mas_lenta']}** es la "
            f"que mas tarda en responder, mientras que "
            f"**{estadisticas['dependencia_mas_rapida']}** es la mas eficiente. "
            f"El tipo de tramite con mayor volumen de radicados es "
            f"**{estadisticas['tipo_mayor_volumen']}**.\n\n"
            "Para el proyecto de Clasificador de PQRS, esto sugiere que un "
            "modelo de clasificacion o priorizacion deberia tomar en cuenta "
            "tanto el tipo de tramite como la dependencia asignada, ya que "
            "ambos factores estan relacionados con el tiempo esperado de "
            "resolucion. Esta informacion puede usarse mas adelante para "
            "entrenar un modelo que prediga el tiempo de respuesta o "
            "priorice automaticamente los tramites con mayor riesgo de "
            "demora.\n"
        )


if __name__ == "__main__":
    registros = leer_datos("data/pqrs_cartago.csv")
    estadisticas = calcular_estadisticas(registros)
    generar_informe(estadisticas, "data/informe_proyecto.md")
    if estadisticas:
        print("Informe generado correctamente: data/informe_proyecto.md")
