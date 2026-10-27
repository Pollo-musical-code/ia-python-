"""
neuronas_logicas.py
Actividad independiente - Clase 8.
Neuronas artificiales (perceptrón) que aprenden las compuertas OR y NOT.
También se prueba AND (para comparar) y XOR (que una sola neurona no puede aprender).
Python puro, sin librerías externas.
"""


def escalon(z):
    """Función de activación escalón."""
    return 1 if z >= 0 else 0


def predecir(entradas, pesos, b):
    """Suma ponderada + bias, pasada por la función escalón."""
    z = sum(w * x for w, x in zip(pesos, entradas)) + b
    return escalon(z)


def entrenar(nombre, datos, pesos, b, tasa=0.1, max_epocas=100):
    """Entrena una neurona y retorna (pesos, b, errores_por_epoca, epocas_necesarias)."""
    print("=" * 60)
    print(f"COMPUERTA {nombre}")
    print("=" * 60)
    print(f"Parámetros iniciales: pesos={pesos}, b={b}, tasa={tasa}")

    errores_por_epoca = []
    epocas_necesarias = None

    for epoca in range(1, max_epocas + 1):
        error_total = 0
        for *entradas, y_real in datos:
            y_pred = predecir(entradas, pesos, b)
            error = y_real - y_pred
            error_total += abs(error)

            # Ajuste de pesos y bias
            pesos = [w + tasa * error * x for w, x in zip(pesos, entradas)]
            b += tasa * error

        errores_por_epoca.append(error_total)
        if error_total == 0:
            epocas_necesarias = epoca
            break

    if len(errores_por_epoca) > 12:
        resumen = f"{errores_por_epoca[:12]} ... (total {len(errores_por_epoca)} épocas)"
    else:
        resumen = errores_por_epoca
    print(f"Error total por época: {resumen}")
    print(f"Parámetros finales: pesos={[round(w, 3) for w in pesos]}, b={b:.3f}")

    if epocas_necesarias:
        print(f"-> Aprendió en {epocas_necesarias} épocas.")
    else:
        print(f"-> NO aprendió: sigue con error tras {max_epocas} épocas.")

    print("Verificación:")
    for *entradas, y_real in datos:
        y_pred = predecir(entradas, pesos, b)
        estado = "OK" if y_pred == y_real else "X"
        print(f"  {estado} Entrada: {tuple(entradas)} | Esperado: {y_real} | Predicho: {y_pred}")
    print()

    return pesos, b, errores_por_epoca, epocas_necesarias


# Datos de entrenamiento: (x1, x2, y) o (x1, y)
DATOS_OR = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
DATOS_NOT = [(0, 1), (1, 0)]
DATOS_AND = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
DATOS_XOR = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

if __name__ == "__main__":
    resultados = {}

    _, _, _, resultados["OR"] = entrenar("OR", DATOS_OR, [0.1, 0.1], 0.0)
    _, _, _, resultados["NOT"] = entrenar("NOT", DATOS_NOT, [0.1], 0.0)
    _, _, _, resultados["AND"] = entrenar("AND (comparación)", DATOS_AND, [0.1, 0.1], 0.0)
    _, _, _, resultados["XOR"] = entrenar("XOR (no aprendible)", DATOS_XOR, [0.1, 0.1], 0.0, max_epocas=100)

    print("=" * 60)
    print("RESUMEN: ÉPOCAS NECESARIAS")
    print("=" * 60)
    for nombre, epocas in resultados.items():
        print(f"  {nombre}: {epocas if epocas else 'no aprendió'}")