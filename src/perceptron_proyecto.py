"""
perceptron_proyecto.py
Avance 2 - Perceptrón desde cero (NumPy, sin Scikit-Learn) aplicado al proyecto PQRS.

Objetivo (y):  demora_alta = 1 si tiempo_respuesta_dias > mediana (6.5 días), 0 si no.
               tiempo_respuesta_dias NO se usa como entrada: con él se calcula el objetivo.
Entradas:      x1 = radicados
               x2 = carga_dependencia (total de radicados que maneja la dependencia)
               Ambas se estandarizan (media 0, desviación 1).

Requiere haber ejecutado antes:  python3 src/preparacion_pqrs.py   (genera data/proyecto_limpio.csv)
Genera: data/curva_aprendizaje_proyecto.png y data/frontera_decision_proyecto.png
Ejecución (desde la raíz del proyecto o desde src/):  python3 perceptron_proyecto.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "proyecto_limpio.csv"
DIR_SALIDA = RAIZ / "data"

COLUMNAS_X = ["radicados", "carga_dependencia"]
COLUMNA_Y = "demora_alta"

TASA_APRENDIZAJE = 0.1
EPOCAS = 500


# ============================================================
# Piezas del perceptrón
# ============================================================
def suma_ponderada(X, w, b):
    """z = w1*x1 + w2*x2 + b para todas las filas a la vez (vectorizado)."""
    return X @ w + b


def sigmoide(z):
    """Convierte cualquier z en un valor entre 0 y 1."""
    return 1 / (1 + np.exp(-z))


def predecir_probabilidad(X, w, b):
    return sigmoide(suma_ponderada(X, w, b))


def calcular_perdida(y_real, y_prob):
    """Entropía cruzada binaria."""
    y_prob = np.clip(y_prob, 1e-9, 1 - 1e-9)
    return -np.mean(y_real * np.log(y_prob) + (1 - y_real) * np.log(1 - y_prob))


def calcular_exactitud(y_real, y_prob):
    return np.mean((y_prob >= 0.5).astype(int) == y_real)


def calcular_gradientes(X, y_real, y_prob):
    """Gradientes de w y b, promediados sobre todos los ejemplos."""
    error = y_prob - y_real
    return X.T @ error / len(y_real), np.mean(error)


def entrenar(X_esc, y):
    """Descenso del gradiente. Retorna (w, b, historial_perdida, historial_exactitud)."""
    w = np.zeros(X_esc.shape[1])
    b = 0.0
    hist_perdida, hist_exactitud = [], []

    for epoca in range(1, EPOCAS + 1):
        y_prob = predecir_probabilidad(X_esc, w, b)
        gradiente_w, gradiente_b = calcular_gradientes(X_esc, y, y_prob)

        w = w - TASA_APRENDIZAJE * gradiente_w
        b = b - TASA_APRENDIZAJE * gradiente_b

        hist_perdida.append(calcular_perdida(y, y_prob))
        hist_exactitud.append(calcular_exactitud(y, y_prob))
        if epoca == 1 or epoca % 100 == 0:
            print(f"Época {epoca:>3} | Pérdida: {hist_perdida[-1]:.4f} | "
                  f"Exactitud: {hist_exactitud[-1]:.2%} | w=[{w[0]:.3f}, {w[1]:.3f}] b={b:.3f}")
    return w, b, hist_perdida, hist_exactitud


# ============================================================
# Gráficos
# ============================================================
def graficar_curva(hist_perdida, hist_exactitud):
    fig, (eje1, eje2) = plt.subplots(1, 2, figsize=(12, 4.5))
    eje1.plot(hist_perdida, color="red")
    eje1.set_title("Pérdida por época")
    eje1.set_xlabel("Época")
    eje1.grid(True, alpha=0.3)
    eje2.plot(hist_exactitud, color="green")
    eje2.set_title("Exactitud por época")
    eje2.set_xlabel("Época")
    eje2.set_ylim(0, 1.05)
    eje2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIR_SALIDA / "curva_aprendizaje_proyecto.png", dpi=150)
    plt.close()


def graficar_frontera(X, y, w, b, media, desviacion):
    x1_original = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x1_esc = (x1_original - media[0]) / desviacion[0]
    # Frontera: z = 0  ->  w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1 + b) / w2
    x2_esc = -(w[0] * x1_esc + b) / w[1]
    x2_original = x2_esc * desviacion[1] + media[1]

    plt.figure(figsize=(9, 6))
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="red", edgecolors="black", s=110, label="Demora alta (1)")
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="green", marker="s", edgecolors="black", s=110, label="Demora no alta (0)")
    plt.plot(x1_original, x2_original, color="blue", linewidth=2.5, label="Frontera de decisión")
    plt.ylim(X[:, 1].min() - 20, X[:, 1].max() + 20)
    plt.title("Frontera de decisión – PQRS Cartago")
    plt.xlabel("Radicados")
    plt.ylabel("Carga de la dependencia (total de radicados)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIR_SALIDA / "frontera_decision_proyecto.png", dpi=150)
    plt.close()


# ============================================================
# Programa principal
# ============================================================
def main():
    print("=" * 60)
    print("PERCEPTRÓN DESDE CERO - PROYECTO PQRS CARTAGO")
    print("=" * 60)

    try:
        df = pd.read_csv(RUTA_DATOS)
    except FileNotFoundError:
        print(f"No existe {RUTA_DATOS}. Ejecute antes: python3 src/preparacion_pqrs.py")
        return

    X = df[COLUMNAS_X].to_numpy(dtype=float)
    y = df[COLUMNA_Y].to_numpy()
    print(f"Datos: X {X.shape} | y {y.shape} | positivos (demora alta): {y.sum()} de {len(y)}\n")

    # Estandarización: se guardan media y desviación para usarlas con casos nuevos
    media = X.mean(axis=0)
    desviacion = X.std(axis=0)
    X_esc = (X - media) / desviacion

    w, b, hist_perdida, hist_exactitud = entrenar(X_esc, y)

    y_prob = predecir_probabilidad(X_esc, w, b)
    print(f"\nRESULTADOS")
    print(f"  Pérdida: {hist_perdida[0]:.4f} (inicial) -> {calcular_perdida(y, y_prob):.4f} (final)")
    print(f"  Exactitud final: {calcular_exactitud(y, y_prob):.2%}")
    print(f"  Pesos: w1 (radicados) = {w[0]:.4f} | w2 (carga_dependencia) = {w[1]:.4f} | b = {b:.4f}")

    graficar_curva(hist_perdida, hist_exactitud)
    graficar_frontera(X, y, w, b, media, desviacion)
    print("\nGráficos guardados: curva_aprendizaje_proyecto.png y frontera_decision_proyecto.png (en data/)")

    # Casos nuevos: misma preparación que en el entrenamiento (media y desviación de los datos originales)
    print("\nCASOS NUEVOS")
    casos = [
        ("Planeación", 45, 192),
        ("Obras Públicas", 33, 97),
        ("Hacienda", 27, 79),
    ]
    for dependencia, radicados, carga in casos:
        caso_esc = (np.array([[radicados, carga]], dtype=float) - media) / desviacion
        p = predecir_probabilidad(caso_esc, w, b)[0]
        decision = "DEMORA ALTA" if p >= 0.5 else "demora no alta"
        print(f"  {dependencia:<15} | radicados={radicados:<3} carga={carga:<3} -> {p:.2%} -> {decision}")


if __name__ == "__main__":
    main()