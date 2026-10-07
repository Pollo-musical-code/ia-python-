"""
preparacion_pqrs.py
Clase 7: Pandas, Seaborn y preparación de datos para Machine Learning.

Carga el CSV de PQRS de Cartago, lo explora, lo limpia, genera
visualizaciones con Seaborn y lo deja listo para un futuro modelo.

Ubicación sugerida: src/preparacion_pqrs.py
Ejecución (desde la raíz del proyecto, dentro del contenedor):
    python src/preparacion_pqrs.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventanas: funciona dentro de Docker
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CSV = RAIZ / "data" / "pqrs_cartago.csv"
DIR_SALIDA = RAIZ / "data"

sns.set_theme(style="whitegrid", context="talk")


def cargar_datos(ruta_csv):
    """Carga el CSV y retorna un DataFrame (o None si no existe)."""
    try:
        df = pd.read_csv(ruta_csv)
        print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
        return df
    except FileNotFoundError:
        print(f"Error: el archivo '{ruta_csv}' no existe.")
        return None


def explorar_datos(df):
    """Muestra head, info, describe y nulos por columna."""
    print("\nPRIMERAS 5 FILAS:")
    print(df.head())

    print("\nINFORMACIÓN GENERAL:")
    df.info()

    print("\nESTADÍSTICAS DESCRIPTIVAS:")
    print(df.describe())

    print("\nVALORES NULOS POR COLUMNA:")
    print(df.isnull().sum())

    # Hallazgo para documentar en el README
    prom_tipo = df.groupby("tipo")["tiempo_respuesta_dias"].mean().sort_values()
    print("\nTIEMPO PROMEDIO POR TIPO (días):")
    print(prom_tipo.round(2))
    print(f"-> Más lento: {prom_tipo.idxmax()} | Más rápido: {prom_tipo.idxmin()}")


def clasificar_riesgo(dias):
    """Convierte los días de respuesta en un nivel de riesgo de demora."""
    if dias <= 5:
        return "bajo"
    if dias <= 8:
        return "medio"
    return "alto"


def limpiar_datos(df):
    """Maneja nulos y duplicados, y crea la columna derivada riesgo_demora."""
    df_limpio = df.copy()

    # Duplicados exactos
    antes = len(df_limpio)
    df_limpio = df_limpio.drop_duplicates()
    print(f"\nDuplicados eliminados: {antes - len(df_limpio)}")

    # Nulos numéricos -> mediana (robusta ante valores extremos)
    for col in df_limpio.select_dtypes(include="number").columns:
        nulos = df_limpio[col].isnull().sum()
        if nulos:
            mediana = df_limpio[col].median()
            df_limpio[col] = df_limpio[col].fillna(mediana)
            print(f"'{col}': {nulos} nulos imputados con la mediana ({mediana}).")

    # Nulos categóricos -> moda
    for col in df_limpio.select_dtypes(exclude="number").columns:
        nulos = df_limpio[col].isnull().sum()
        if nulos:
            moda = df_limpio[col].mode()[0]
            df_limpio[col] = df_limpio[col].fillna(moda)
            print(f"'{col}': {nulos} nulos imputados con la moda ({moda}).")

    if df_limpio.isnull().sum().sum() == 0:
        print("Sin valores nulos pendientes.")

    # Columna derivada: nivel de riesgo (base del futuro clasificador)
    df_limpio["riesgo_demora"] = df_limpio["tiempo_respuesta_dias"].apply(
        clasificar_riesgo
    )
    print("Columna 'riesgo_demora' creada (bajo <=5, medio <=8, alto >8 días).")
    print(df_limpio["riesgo_demora"].value_counts())

    return df_limpio


def visualizar_datos(df):
    """Genera 4 gráficos con Seaborn y los guarda en data/."""
    # 1. Histograma / densidad del tiempo de respuesta
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="tiempo_respuesta_dias", bins=6, kde=True, color="skyblue")
    plt.title("Distribución del tiempo de respuesta")
    plt.xlabel("Días")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(DIR_SALIDA / "sns_histograma_tiempo.png", dpi=150)
    plt.close()

    # 2. Barras: tiempo promedio por tipo de trámite
    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=df,
        x="tipo",
        y="tiempo_respuesta_dias",
        hue="tipo",
        errorbar=None,
        palette="viridis",
        legend=False,
    )
    plt.title("Tiempo promedio de respuesta por tipo")
    plt.xlabel("Tipo de trámite")
    plt.ylabel("Días promedio")
    plt.tight_layout()
    plt.savefig(DIR_SALIDA / "sns_barras_tipo.png", dpi=150)
    plt.close()

    # 3. Boxplot: distribución por dependencia
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="dependencia", y="tiempo_respuesta_dias")
    plt.title("Tiempo de respuesta por dependencia")
    plt.xlabel("Dependencia")
    plt.ylabel("Días")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(DIR_SALIDA / "sns_boxplot_dependencia.png", dpi=150)
    plt.close()

    # 4. Dispersión: radicados vs tiempo, con línea de tendencia
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df, x="radicados", y="tiempo_respuesta_dias", hue="tipo", s=120
    )
    sns.regplot(
        data=df,
        x="radicados",
        y="tiempo_respuesta_dias",
        scatter=False,
        color="gray",
        line_kws={"linestyle": "--"},
    )
    plt.title("Radicados vs tiempo de respuesta")
    plt.xlabel("Radicados")
    plt.ylabel("Días")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(DIR_SALIDA / "sns_dispersion_radicados.png", dpi=150)
    plt.close()

    print("\nGráficos guardados en data/ (prefijo sns_).")


def preparar_ml(df):
    """Codifica variables categóricas y deja el DataFrame numérico."""
    df_ml = df.copy()

    # Label Encoding para la variable ordinal
    orden = {"bajo": 0, "medio": 1, "alto": 2}
    df_ml["riesgo_demora"] = df_ml["riesgo_demora"].map(orden)

    # One-Hot Encoding para variables nominales
    df_ml = pd.get_dummies(
        df_ml, columns=["tipo", "dependencia"], prefix=["tipo", "dep"], dtype=int
    )
    print("\nCodificación aplicada: One-Hot (tipo, dependencia) y Label (riesgo_demora).")
    print(df_ml.head())
    return df_ml


def main():
    print("=" * 60)
    print("PREPARACIÓN DE DATOS PARA ML - PQRS CARTAGO")
    print("=" * 60)

    df = cargar_datos(RUTA_CSV)
    if df is None:
        return

    explorar_datos(df)
    df_limpio = limpiar_datos(df)
    visualizar_datos(df_limpio)
    df_ml = preparar_ml(df_limpio)

    salida = DIR_SALIDA / "pqrs_preparadas_ml.csv"
    df_ml.to_csv(salida, index=False)
    print(f"\nDataset preparado guardado en '{salida}'")
    print("Proceso completado exitosamente.")


if __name__ == "__main__":
    main()