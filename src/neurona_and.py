"""
neurona_and.py
Clase 8: neurona artificial (perceptrón) que aprende la compuerta AND.
Python puro para el entrenamiento; Matplotlib solo para graficar el error.
"""

import matplotlib

matplotlib.use("Agg")  # sin ventanas: funciona dentro de Docker
import matplotlib.pyplot as plt


def escalon(z):
    """Función de activación escalón."""
    return 1 if z >= 0 else 0


def neurona(x1, x2, w1, w2, b):
    """Calcula la salida de la neurona."""
    z = w1 * x1 + w2 * x2 + b
    return escalon(z)


# Datos de entrenamiento (compuerta AND): (x1, x2, y_real)
datos = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1),
]

# Parámetros iniciales (valores bajos para ver cómo evoluciona el error)
w1 = 0.1
w2 = 0.1
b = 0.0
tasa_aprendizaje = 0.1
MAX_EPOCAS = 20

errores_por_epoca = []

print("=" * 60)
print("ENTRENAMIENTO DE UNA NEURONA PARA LA COMPUERTA AND")
print("=" * 60)
print(f"Parámetros iniciales: w1={w1}, w2={w2}, b={b}\n")

for epoca in range(MAX_EPOCAS):
    error_total = 0
    print(f"--- Época {epoca + 1} ---")

    for x1, x2, y_real in datos:
        y_pred = neurona(x1, x2, w1, w2, b)
        error = y_real - y_pred
        error_total += abs(error)

        # Ajuste de pesos y bias
        w1 += tasa_aprendizaje * error * x1
        w2 += tasa_aprendizaje * error * x2
        b += tasa_aprendizaje * error

        print(f"  Entrada: ({x1}, {x2}) | Real: {y_real} | Pred: {y_pred} | Error: {error}")

    errores_por_epoca.append(error_total)
    print(f"  Error total: {error_total}")
    print(f"  Parámetros actuales: w1={w1:.3f}, w2={w2:.3f}, b={b:.3f}\n")

    if error_total == 0:
        print(f"La neurona aprendió en {epoca + 1} épocas.\n")
        break

print("=" * 60)
print("PRUEBA FINAL")
print("=" * 60)
for x1, x2, y_real in datos:
    y_pred = neurona(x1, x2, w1, w2, b)
    estado = "OK" if y_pred == y_real else "X"
    print(f"  {estado} Entrada: ({x1}, {x2}) | Esperado: {y_real} | Predicho: {y_pred}")

# Gráfico de la evolución del error
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(errores_por_epoca) + 1), errores_por_epoca, marker="o", color="red")
plt.title("Evolución del Error durante el Entrenamiento (AND)")
plt.xlabel("Época")
plt.ylabel("Error Total")
plt.grid(True)
plt.savefig("evolucion_error.png", dpi=150)
plt.close()
print("\nGráfico guardado: evolucion_error.png")