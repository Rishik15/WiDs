from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = ROOT_DIR / "src" / "Submission"
CHECK_DIR = ROOT_DIR / "src" / "ReproducibilityCheck"

sys.path.append(str(SUBMISSION_DIR))

from src.Submission.submit import generate_submission


def compare_submissions(
    generated_path: Path,
    kaggle_path: Path,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    generated_sub = pd.read_csv(generated_path)
    kaggle_sub = pd.read_csv(kaggle_path)

    id_col = "event_id"
    value_cols = ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]

    generated_s = generated_sub.sort_values(id_col).reset_index(drop=True)
    kaggle_s = kaggle_sub.sort_values(id_col).reset_index(drop=True)

    print("Generated shape:", generated_s.shape)
    print("Kaggle shape:", kaggle_s.shape)

    print("Generated columns:", list(generated_s.columns))
    print("Kaggle columns:", list(kaggle_s.columns))

    same_ids = generated_s[id_col].equals(kaggle_s[id_col])
    exact_same_values = generated_s[value_cols].equals(kaggle_s[value_cols])

    close_values = np.allclose(
        generated_s[value_cols].values,
        kaggle_s[value_cols].values,
        rtol=rtol,
        atol=atol,
    )

    diff = np.abs(generated_s[value_cols].values - kaggle_s[value_cols].values)

    print("Same IDs:", same_ids)
    print("Exact same values:", exact_same_values)
    print("Close values:", close_values)
    print("Max difference:", diff.max())
    print("Mean difference:", diff.mean())

    if not same_ids:
        raise AssertionError(
            "Generated submission and Kaggle submission have different event IDs."
        )

    if not close_values:
        raise AssertionError(
            "Generated submission does not match Kaggle submission within tolerance."
        )

    print("Reproducibility check passed.")


def main() -> None:
    generated_path = CHECK_DIR / "generated_submission.csv"
    kaggle_path = CHECK_DIR / "kaggle_submission.csv"

    print("Regenerating submission...")
    generated_sub = generate_submission(output_path=generated_path)

    generated_sub.to_csv(generated_path, index=False)

    print("Comparing regenerated submission with Kaggle submission...")
    compare_submissions(generated_path, kaggle_path)


if __name__ == "__main__":
    main()
