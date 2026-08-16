import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Configuración
RANDOM_SEED = 0
TEST_SIZE = 0.30

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "data_clientes.xlsx"

# Carga de datos
df = pd.read_excel(DATA_PATH)

print(f"Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas")

# Definición de variables
TARGET = "Default"
DROP_COLUMNS = [
    "Id_Cliente",
    "ID",
    "id",
    "id_cliente",
]

DROP_COLUMNS = [column for column in DROP_COLUMNS if column in df.columns]

FEATURE_COLUMNS = [
    "Edad",
    "Nivel_Educacional",
    "Años_Trabajando",
    "Ingresos",
    "Deuda_Comercial",
    "Deuda_Credito",
    "Otras_Deudas",
]

df_model = df.drop_duplicates(subset=FEATURE_COLUMNS).copy()

print(f"Filas originales: {len(df)}")
print(f"Filas después de eliminar duplicados: {len(df_model)}")
print(f"Filas eliminadas: {len(df) - len(df_model)}")

X = df_model[FEATURE_COLUMNS]
y = df_model[TARGET]

print("\nDistribución de Default después de eliminar duplicados:")
print(y.value_counts())
print(y.value_counts(normalize=True))

# División Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_SEED,
)


print(f"Datos de entrenamiento: {X_train.shape[0]}")
print(f"Datos de prueba: {X_test.shape[0]}")
print(f"Features utilizadas: {list(X.columns)}")

NUMERIC_FEATURES = [
    "Edad",
    "Años_Trabajando",
    "Ingresos",
    "Deuda_Comercial",
    "Deuda_Credito",
    "Otras_Deudas",
]

CATEGORICAL_FEATURES = [
    "Nivel_Educacional",
]

FEATURE_TYPES = {
    "Edad": "integer",
    "Nivel_Educacional": "string",
    "Años_Trabajando": "integer",
    "Ingresos": "integer",
    "Deuda_Comercial": "number",
    "Deuda_Credito": "number",
    "Otras_Deudas": "number",
}

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)

model = HistGradientBoostingClassifier(random_state=RANDOM_SEED)

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

model_pipeline.fit(X_train, y_train)

print("Modelo entrenado correctamente.")

# evaluación sobre datos no visto
y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nMétricas sobre datos de prueba:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1:      {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# guardar el modelo
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.joblib"

joblib.dump(model_pipeline, MODEL_PATH)

print(f"\nModelo guardado en: {MODEL_PATH}")

METADATA_PATH = MODEL_DIR / "metadata.json"

metadata = {
    "model": "HistGradientBoostingClassifier",
    "python_version": platform.python_version(),
    "sklearn_version": sklearn.__version__,
    "training_timestamp": datetime.now(timezone.utc).isoformat(),
    "random_seed": RANDOM_SEED,
    "test_size": TEST_SIZE,
    "features": list(X.columns),
    "feature_types": FEATURE_TYPES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "metrics": {
        "accuracy": round(float(accuracy), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
    },
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2, ensure_ascii=False)

print(f"Metadata guardada en: {METADATA_PATH}")
