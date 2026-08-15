import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

def validate_metrics(
    metrics: dict,
    min_f1: float,
    min_roc_auc: float,
) -> list[str]:
    failures = []

    if metrics["f1"] < min_f1:
        failures.append(
            f"F1 {metrics['f1']:.4f} < mínimo {min_f1:.4f}"
        )

    if metrics["roc_auc"] < min_roc_auc:
        failures.append(
            "ROC-AUC "
            f"{metrics['roc_auc']:.4f} < mínimo {min_roc_auc:.4f}"
        )

    return failures

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida las métricas del modelo."
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--min-roc-auc",
        type=float,
        default=0.75,
    )

    args = parser.parse_args()

    if not METADATA_PATH.exists():
        print(f"ERROR: no se encontró {METADATA_PATH}")
        return 2

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )
    metrics = metadata["metrics"]

    print("=== Quality Gate del Modelo ===")
    print(f"F1:      {metrics['f1']:.4f} (mínimo {args.min_f1:.4f})")
    print(
        "ROC-AUC: "
        f"{metrics['roc_auc']:.4f} "
        f"(mínimo {args.min_roc_auc:.4f})"
    )

    failures = validate_metrics(
        metrics=metrics,
        min_f1=args.min_f1,
        min_roc_auc=args.min_roc_auc,
    )

    if failures:
        print("\nGATE FALLIDO:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nGATE APROBADO: el modelo cumple los umbrales.")
    return 0

if __name__ == "__main__":
    sys.exit(main())