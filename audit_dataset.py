from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "data_clientes.xlsx"

TARGET = "Default"

df = pd.read_excel(DATA_PATH)

print("=" * 60)
print("AUDITORÍA DEL DATASET")
print("=" * 60)

print(f"\nFilas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")

print("\nColumnas:")
for column in df.columns:
    print(f"- {column}")

print("\nTipos de datos:")
print(df.dtypes)

print("\nValores nulos:")
print(df.isnull().sum())

print("\nDistribución del target:")
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True))

print("\nValores únicos por columna:")
for column in df.columns:
    print(f"{column}: {df[column].nunique()}")

print("\n" + "=" * 60)
print("AUDITORÍA DE DUPLICADOS")
print("=" * 60)

FEATURE_COLUMNS = [
    "Edad",
    "Nivel_Educacional",
    "Años_Trabajando",
    "Ingresos",
    "Deuda_Comercial",
    "Deuda_Credito",
    "Otras_Deudas",
]

# Duplicados considerando features + target
duplicates = df.duplicated(subset=FEATURE_COLUMNS + [TARGET]).sum()

print("\nDuplicados en features + target:")
print(duplicates)

# Duplicados considerando solamente las features
duplicates_features = df.duplicated(subset=FEATURE_COLUMNS).sum()

print("\nDuplicados considerando solamente las features:")
print(duplicates_features)

# Casos donde las mismas features tienen distintos targets
target_variation = df.groupby(FEATURE_COLUMNS)[TARGET].nunique()

conflicting_rows = (target_variation > 1).sum()

print("\nCombinaciones de features con distintos valores de Default:")
print(conflicting_rows)

print("\n" + "=" * 60)
print("FRECUENCIA DE COMBINACIONES DE FEATURES")
print("=" * 60)

feature_counts = (
    df.groupby(FEATURE_COLUMNS, dropna=False).size().sort_values(ascending=False)
)

print("\nCantidad de combinaciones únicas:")
print(len(feature_counts))

print("\nCombinaciones más repetidas:")
print(feature_counts.head(10))

print("\n" + "=" * 60)
print("DISTRIBUCIÓN DE FRECUENCIA DE LAS COMBINACIONES")
print("=" * 60)

feature_counts = df.groupby(FEATURE_COLUMNS, dropna=False).size()

print(f"\nCombinaciones únicas: {len(feature_counts)}")
print(f"Filas totales: {len(df)}")

print("\nFrecuencia de las combinaciones:")

print(feature_counts.value_counts().sort_index())
