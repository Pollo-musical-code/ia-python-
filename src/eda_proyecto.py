"""
eda_proyecto.py
Analisis Exploratorio de Datos (EDA) del proyecto Clasificador de PQRS,
utilizando NumPy y Matplotlib. Actividad independiente - Clase 4.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv


def cargar_datos(archivo_csv):
    """
    Carga el archivo CSV de PQRS y retorna un array de NumPy con las
    columnas numericas (radicados y tiempo_respuesta_dias), la lista
    completa de datos, y los nombres de columnas.
    """
    datos = []
    nombres_columnas = []
    try:
        with open(archivo_csv, "r", encoding="utf-8") as archivo:
            lector = csv.reader(archivo)
            nombres_columnas = next(lector)  # Leer encabezados

            for fila in lector:
                # fila = [tipo, radicados, tiempo_respuesta_dias, dependencia]
                radicados = float(fila[1])
                tiempo_respuesta = float(fila[2])
                datos.append([fila[0], radicados, tiempo_respuesta, fila[3]])

        print(f"Datos cargados correctamente: {len(datos)} registros.")
    except FileNotFoundError:
        print(f"Error: el archivo '{archivo_csv}' no existe.")
        return None, None, None
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None, None, None

    # Columnas numericas: radicados (indice 1) y tiempo_respuesta (indice 2)
    datos_numericos = np.array([[fila[1], fila[2]] for fila in datos], dtype=float)

    return datos_numericos, datos, nombres_columnas


def analizar_datos(datos_numericos):
    """Calcula estadisticas descriptivas de radicados y tiempo de respuesta."""
    if datos_numericos is None or len(datos_numericos) == 0:
        return None

    radicados = datos_numericos[:, 0]
    tiempo_respuesta = datos_numericos[:, 1]

    estadisticas = {
        "radicados": {
            "media": np.mean(radicados),
            "mediana": np.median(radicados),
            "desviacion": np.std(radicados),
            "minimo": np.min(radicados),
            "maximo": np.max(radicados),
        },
        "tiempo_respuesta_dias": {
            "media": np.mean(tiempo_respuesta),
            "mediana": np.median(tiempo_respuesta),
            "desviacion": np.std(tiempo_respuesta),
            "minimo": np.min(tiempo_respuesta),
            "maximo": np.max(tiempo_respuesta),
        },
    }
    return estadisticas


def generar_visualizaciones(datos_numericos):
    """Genera un histograma y un grafico de dispersion, guardados como PNG."""
    if datos_numericos is None or len(datos_numericos) == 0:
        return

    radicados = datos_numericos[:, 0]
    tiempo_respuesta = datos_numericos[:, 1]

    # 1. Histograma: distribucion del tiempo de respuesta
    plt.figure(figsize=(8, 5))
    plt.hist(tiempo_respuesta, bins=5, color="orange", edgecolor="black")
    plt.title("Distribucion del tiempo de respuesta (PQRS)")
    plt.xlabel("Tiempo de respuesta (dias)")
    plt.ylabel("Frecuencia")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("data/eda_histograma_tiempo_respuesta.png")
    plt.close()

    # 2. Dispersion: relacion entre radicados y tiempo de respuesta
    plt.figure(figsize=(8, 6))
    plt.scatter(radicados, tiempo_respuesta, color="green", alpha=0.7)
    plt.title("Relacion: Radicados vs Tiempo de respuesta")
    plt.xlabel("Radicados")
    plt.ylabel("Tiempo de respuesta (dias)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/eda_dispersion_radicados_tiempo.png")
    plt.close()

    print("Graficos guardados en data/: eda_histograma_tiempo_respuesta.png y eda_dispersion_radicados_tiempo.png")


def main():
    print("=" * 50)
    print(" ANALISIS EXPLORATORIO DE DATOS (EDA) - PROYECTO PQRS")
    print("=" * 50)

    datos_numericos, datos_completos, columnas = cargar_datos("data/pqrs_cartago.csv")
    if datos_numericos is None:
        return

    estadisticas = analizar_datos(datos_numericos)
    if estadisticas:
        print("\nESTADISTICAS DESCRIPTIVAS:")
        print("-" * 40)
        for variable, valores in estadisticas.items():
            print(f"\n{variable.upper()}:")
            for key, value in valores.items():
                print(f"  {key.capitalize()}: {value:.2f}")

    print("\nGENERANDO VISUALIZACIONES...")
    generar_visualizaciones(datos_numericos)

    print("\nAnalisis completado exitosamente.")


if __name__ == "__main__":
    main()
