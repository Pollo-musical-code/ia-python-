"""
neurona_pqrs.py
Clase 8 aplicada al proyecto: una neurona artificial (perceptrón) que aprende
a predecir si un trámite PQRS tendrá RIESGO DE DEMORA ALTO.

Entradas (lo que se conoce al radicar el trámite):
    x1 = radicados (normalizado entre 0 y 1)
    x2 = 1 si el tipo es Queja, 0 si no
    x3 = 1 si el tipo es Reclamo, 0 si no
Salida esperada (y):
    1 = riesgo alto (más de 8 días de respuesta), 0 = no es riesgo alto

El tiempo de respuesta NO se usa como entrada: con él se calcula la salida
esperada, y usarlo sería darle la respuesta a la neurona.
Entrenamiento en Python puro, como en la clase 8. Matplotlib solo para graficar.
"""

import csv
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CSV = RAIZ / "data" / "pqrs_cartago.csv"
RUTA_GRAFICO = RAIZ / "data" / "neurona_pqrs_error.png"

UMBRAL_RIESGO_ALTO = 8  # días: más de 8 = riesgo alto (igual que en riesgo_demora)


# ============================================================
# 1. Función de activación y neurona (como en la clase 8)
# ============================================================
def escalon(z):
    """Función de activación escalón."""
    return 1 if z >= 0 else 0


def predecir(entradas, pesos, b):
    """z = suma(w * x) + b, pasada por la función escalón."""
    z = sum(w * x for w, x in zip(pesos, entradas)) + b
    return escalon(z)


# ============================================================
# 2. Datos: del CSV a entradas y salidas
# ============================================================
def leer_registros(ruta_csv):
    """Lee el CSV (como en la clase 3) y retorna una lista de diccionarios."""
    registros = []
    try:
        with open(ruta_csv, "r", encoding="utf-8") as archivo:
            for fila in csv.DictReader(archivo):
                registros.append(
                    {
                        "tipo": fila["tipo"],
                        "radicados": int(fila["radicados"]),
                        "tiempo": int(fila["tiempo_respuesta_dias"]),
                    }
                )
    except FileNotFoundError:
        print(f"Error: no se encontró '{ruta_csv}'.")
    return registros


def codificar(tipo, radicados, rad_min, rad_max):
    """Convierte un trámite en el vector de entradas [x1, x2, x3]."""
    rad_norm = (radicados - rad_min) / (rad_max - rad_min)
    return [rad_norm, 1 if tipo == "Queja" else 0, 1 if tipo == "Reclamo" else 0]


def preparar_datos(registros):
    """Retorna (datos, rad_min, rad_max); datos = lista de (entradas, y_real)."""
    radicados = [r["radicados"] for r in registros]
    rad_min, rad_max = min(radicados), max(radicados)
    datos = []
    for r in registros:
        entradas = codificar(r["tipo"], r["radicados"], rad_min, rad_max)
        y_real = 1 if r["tiempo"] > UMBRAL_RIESGO_ALTO else 0
        datos.append((entradas, y_real))
    return datos, rad_min, rad_max


# ============================================================
# 3. Entrenamiento
# ============================================================
def entrenar(datos, tasa=0.1, max_epocas=200):
    """Entrena la neurona. Retorna (pesos, b, errores_por_epoca, epocas)."""
    pesos = [0.0] * len(datos[0][0])
    b = 0.0
    errores_por_epoca = []
    epocas = None

    print(f"Parámetros iniciales: pesos={pesos}, b={b}, tasa={tasa}\n")

    for epoca in range(1, max_epocas + 1):
        error_total = 0
        for entradas, y_real in datos:
            y_pred = predecir(entradas, pesos, b)
            error = y_real - y_pred
            error_total += abs(error)

            # Ajuste de pesos y bias
            pesos = [w + tasa * error * x for w, x in zip(pesos, entradas)]
            b += tasa * error

        errores_por_epoca.append(error_total)
        if epoca <= 5 or error_total == 0:
            print(f"Época {epoca:>3} | Error total: {error_total}")
        if error_total == 0:
            epocas = epoca
            break

    return pesos, b, errores_por_epoca, epocas


# ============================================================
# 4. Gráfico del error
# ============================================================
def graficar_error(errores_por_epoca):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib no está instalado: se omite el gráfico.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(errores_por_epoca) + 1), errores_por_epoca, marker="o", color="red")
    plt.title("Evolución del error: neurona de riesgo de demora (PQRS)")
    plt.xlabel("Época")
    plt.ylabel("Error total")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RUTA_GRAFICO, dpi=150)
    plt.close()
    print(f"Gráfico guardado: {RUTA_GRAFICO}")


# ============================================================
# 5. Programa principal
# ============================================================
def main():
    print("=" * 60)
    print("NEURONA ARTIFICIAL: RIESGO DE DEMORA ALTO EN PQRS")
    print("=" * 60)

    registros = leer_registros(RUTA_CSV)
    if not registros:
        return

    datos, rad_min, rad_max = preparar_datos(registros)
    print(f"{len(datos)} trámites | riesgo alto: {sum(y for _, y in datos)}\n")

    pesos, b, errores, epocas = entrenar(datos)

    if epocas:
        print(f"\nLa neurona aprendió en {epocas} épocas.")
    else:
        print("\nLa neurona NO llegó a error 0 (los datos podrían no ser separables).")

    print("\nPARÁMETROS APRENDIDOS")
    for nombre, w in zip(["radicados", "es_queja", "es_reclamo"], pesos):
        print(f"  w({nombre}) = {w:.3f}")
    print(f"  b = {b:.3f}")

    print("\nVERIFICACIÓN")
    aciertos = 0
    for r, (entradas, y_real) in zip(registros, datos):
        y_pred = predecir(entradas, pesos, b)
        aciertos += y_pred == y_real
        estado = "OK" if y_pred == y_real else "X "
        print(
            f"  {estado} {r['tipo']:<10} radicados={r['radicados']:<3} "
            f"tiempo={r['tiempo']:<2} | esperado={y_real} predicho={y_pred}"
        )
    print(f"Aciertos: {aciertos}/{len(datos)}")

    print("\nPREDICCIONES CON TRÁMITES NUEVOS")
    for tipo, rad in [("Reclamo", 30), ("Petición", 40), ("Queja", 38), ("Queja", 28)]:
        entradas = codificar(tipo, rad, rad_min, rad_max)
        y_pred = predecir(entradas, pesos, b)
        etiqueta = "RIESGO ALTO" if y_pred else "no es riesgo alto"
        print(f"  {tipo} con {rad} radicados -> {etiqueta}")

    graficar_error(errores)


if __name__ == "__main__":
    main()