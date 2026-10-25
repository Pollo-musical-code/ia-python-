"""
preparacion_pqrs.py
Clase 7 + Avance 2: Pandas, Seaborn y preparación de datos para Machine Learning.

Carga el CSV de PQRS de Cartago, lo inspecciona, lo limpia (con tabla antes/después),
crea las variables del modelo, genera 5 gráficos con Seaborn y exporta:
    data/proyecto_limpio.csv        -> dataset limpio con las variables del modelo
    data/pqrs_preparadas_ml.csv     -> dataset codificado (get_dummies)

Ejecución (desde la raíz del proyecto):
    python3 src/preparacion_pqrs.py
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
        return pd.read_csv(ruta_csv)
    except FileNotFoundError:
        print(f"Error: el archivo '{ruta_csv}' no existe.")
        return None


def inspeccionar(df):
    """Muestra shape, info, nulos por columna y duplicados."""
    print(f"\nSHAPE: {df.shape[0]} filas x {df.shape[1]} columnas")
    print("\nINFO:")
    df.info()
    print("\nNULOS POR COLUMNA:")
    print(df.isnull().sum())
    print(f"\nDUPLICADOS: {df.duplicated().sum()}")
    print("\nESTADÍSTICAS DESCRIPTIVAS:")
    print(df.describe())


def diagnosticar(df):
    """Revisa tipos, atípicos (IQR), valores imposibles y categorías inconsistentes."""
    print("\nDIAGNÓSTICO")
    print("-" * 50)

    for col in ["radicados", "tiempo_respuesta_dias"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        atipicos = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        imposibles = (df[col] <= 0).sum()
        print(f"{col}: rango {df[col].min()}-{df[col].max()} | atípicos (IQR): {atipicos} | valores <= 0: {imposibles}")

    for col in ["tipo", "dependencia"]:
        con_espacios = (df[col] != df[col].str.strip()).sum()
        mal_escritas = (df[col] != df[col].str.strip().str.title()).sum()
        print(f"{col}: {sorted(df[col].unique())} | con espacios: {con_espacios} | mayúsculas inconsistentes: {mal_escritas}")


def limpiar(df):
    """Aplica la limpieza: texto, duplicados, imposibles y nulos."""
    df_l = df.copy()

    # Categorías inconsistentes: sin espacios sobrantes y con mayúsculas uniformes
    for col in ["tipo", "dependencia"]:
        df_l[col] = df_l[col].str.strip().str.title()

    # Duplicados exactos
    antes = len(df_l)
    df_l = df_l.drop_duplicates()
    print(f"\nDuplicados eliminados: {antes - len(df_l)}")

    # Valores imposibles: radicados o tiempos <= 0
    antes = len(df_l)
    df_l = df_l[(df_l["radicados"] > 0) & (df_l["tiempo_respuesta_dias"] > 0)]
    print(f"Filas con valores imposibles eliminadas: {antes - len(df_l)}")

    # Nulos: mediana en numéricas, moda en categóricas
    for col in df_l.columns:
        nulos = df_l[col].isnull().sum()
        if nulos:
            if pd.api.types.is_numeric_dtype(df_l[col]):
                df_l[col] = df_l[col].fillna(df_l[col].median())
            else:
                df_l[col] = df_l[col].fillna(df_l[col].mode()[0])
            print(f"'{col}': {nulos} nulos imputados.")
    print(f"Nulos pendientes: {df_l.isnull().sum().sum()}")

    return df_l.reset_index(drop=True)


def clasificar_riesgo(dias):
    """Nivel de riesgo de demora en 3 categorías."""
    if dias <= 5:
        return "bajo"
    if dias <= 8:
        return "medio"
    return "alto"


def crear_variables(df):
    """Crea las variables del modelo: carga_dependencia, demora_alta y riesgo_demora."""
    df_v = df.copy()

    # Entrada 2 del perceptrón: total de radicados que maneja la dependencia
    df_v["carga_dependencia"] = df_v.groupby("dependencia")["radicados"].transform("sum")

    # Objetivo 0/1: 1 si el tiempo de respuesta supera la mediana
    mediana = df_v["tiempo_respuesta_dias"].median()
    df_v["demora_alta"] = (df_v["tiempo_respuesta_dias"] > mediana).astype(int)
    print(f"\nObjetivo 'demora_alta' = 1 si tiempo_respuesta_dias > {mediana} (mediana)")
    print(df_v["demora_alta"].value_counts().sort_index().to_string())
    print("(tiempo_respuesta_dias NO se usa como entrada: con él se calcula el objetivo)")

    # Categoría de 3 niveles (clase 7)
    df_v["riesgo_demora"] = df_v["tiempo_respuesta_dias"].apply(clasificar_riesgo)
    return df_v


def tabla_antes_despues(df_antes, df_despues):
    """Compara filas, columnas, nulos y duplicados antes y después."""
    tabla = pd.DataFrame(
        {
            "Antes": [df_antes.shape[0], df_antes.shape[1], int(df_antes.isnull().sum().sum()), int(df_antes.duplicated().sum())],
            "Después": [df_despues.shape[0], df_despues.shape[1], int(df_despues.isnull().sum().sum()), int(df_despues.duplicated().sum())],
        },
        index=["Filas", "Columnas", "Nulos totales", "Duplicados"],
    )
    print("\nTABLA ANTES / DESPUÉS")
    print(tabla)
    return tabla


def visualizar(df):
    """Genera 5 gráficos con Seaborn y los guarda en data/."""
    mediana = df["tiempo_respuesta_dias"].median()

    # 1. Histograma del tiempo de respuesta
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="tiempo_respuesta_dias", bins=6, kde=True, color="skyblue")
    plt.axvline(mediana, color="red", linestyle="--", label=f"Mediana = {mediana}")
    plt.title("Distribución del tiempo de respuesta")
    plt.xlabel("Días")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DIR_SALIDA / "sns_histograma_tiempo.png", dpi=150)
    plt.close()

    # 2. Barras: tiempo promedio por tipo de trámite
    plt.figure(figsize=(9, 6))
    sns.barplot(data=df, x="tipo", y="tiempo_respuesta_dias", hue="tipo",
                errorbar=None, palette="viridis", legend=False)
    plt.title("Tiempo promedio de respuesta por tipo")
    plt.xlabel("Tipo de trámite")
    plt.ylabel("Días promedio")
    plt.tight_layout()
    plt.savefig(DIR_SALIDA / "sns_barras_tipo.png", dpi=150)
    plt.close()

    # 3. Boxplot por dependencia
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="dependencia", y="tiempo_respuesta_dias")
    plt.title("Tiempo de respuesta por dependencia")
    plt.xlabel("Dependencia")
    plt.ylabel("Días")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(DIR_SALIDA / "sns_boxplot_dependencia.png", dpi=150)
    plt.close()

    # 4. Dispersión: radicados vs tiempo
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="radicados", y="tiempo_respuesta_dias", hue="tipo", s=120)
    sns.regplot(data=df, x="radicados", y="tiempo_respuesta_dias", scatter=False,
                color="gray", line_kws={"linestyle": "--"})
    plt.title("Radicados vs tiempo de respuesta")
    plt.xlabel("Radicados")
    plt.ylabel("Días")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(DIR_SALIDA / "sns_dispersion_radicados.png", dpi=150)
    plt.close()

    # 5. Entradas del modelo vs objetivo (demora_alta)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="radicados", y="carga_dependencia", hue="demora_alta",
                    style="dependencia", palette={0: "tab:green", 1: "tab:red"}, s=170)
    plt.title("Entradas del modelo según la demora (0 = no, 1 = alta)")
    plt.xlabel("Radicados")
    plt.ylabel("Carga de la dependencia (total de radicados)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.savefig(DIR_SALIDA / "sns_entradas_vs_objetivo.png", dpi=150)
    plt.close()

    print("\nGráficos guardados en data/ (prefijo sns_).")


def preparar_ml(df):
    """Codifica las variables categóricas (get_dummies) y guarda el CSV."""
    df_ml = df.copy()
    df_ml["riesgo_demora"] = df_ml["riesgo_demora"].map({"bajo": 0, "medio": 1, "alto": 2})
    df_ml = pd.get_dummies(df_ml, columns=["tipo", "dependencia"], prefix=["tipo", "dep"], dtype=int)
    df_ml.to_csv(DIR_SALIDA / "pqrs_preparadas_ml.csv", index=False)
    print(f"Dataset codificado guardado: data/pqrs_preparadas_ml.csv ({df_ml.shape[1]} columnas)")


def main():
    print("=" * 60)
    print("PREPARACIÓN DE DATOS - PQRS CARTAGO (AVANCE 2)")
    print("=" * 60)

    df = cargar_datos(RUTA_CSV)
    if df is None:
        return

    inspeccionar(df)
    diagnosticar(df)
    df_limpio = limpiar(df)
    df_limpio = crear_variables(df_limpio)
    tabla_antes_despues(df, df_limpio)

    df_limpio.to_csv(DIR_SALIDA / "proyecto_limpio.csv", index=False)
    print("\nDataset limpio guardado: data/proyecto_limpio.csv")

    visualizar(df_limpio)
    preparar_ml(df_limpio)
    print("\nProceso completado exitosamente.")


if __name__ == "__main__":
    main()