# WiDS Global Datathon 2026 — BTT_Heatwave 3rd Place Solution

This repository contains the final solution for team **BTT_Heatwave** for the **WiDS Global Datathon 2026** competition.

Our team placed **3rd** with a score of **0.97250**.

## Team

**Team Name:** BTT_Heatwave

**Team Members:**

- Ayooluwa Wojuade
- Rishik Yesgari

## Competition Overview

The goal of the competition was to predict when a wildfire may threaten an evacuation zone. For each fire, the task was to estimate the probability of a threat within four time horizons:

- 12 hours
- 24 hours
- 48 hours
- 72 hours

The model used only the first five hours of fire perimeter information. This made the task a right-censored time-to-event prediction problem, because some fires reached an evacuation zone while others did not during the observed time window.

## Solution Summary

Our final solution used a horizon-specific ensemble to predict cumulative threat probabilities at 12, 24, 48, and 72 hours.

The pipeline builds 87 engineered features from the provided fire perimeter data. These features describe distance to evacuation zones, movement, fire growth, directional alignment, temporal information, and estimated time-to-arrival.

The final model combines:

- LightGBM models
- XGBoost models
- CatBoost models
- XGBoost AFT survival modeling
- Cox-style survival modeling experiments
- Saved blending and post-processing settings from `src/models/stack_bundle.joblib`

The final Python script retrains the model components, loads the saved blending settings, generates predictions, and writes the final submission file.

In our tested environment, the full submission script completed in approximately **99.40 seconds**, or about **1 minute 39 seconds**.

## Repository Structure

```text
.
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── metaData.csv
│   └── README.md
│
├── src/
│   ├── EDA/
│   │   ├── dataCleaing.ipynb
│   │   └── targetAnalysis.ipynb
│   │
│   ├── Modeling/
│   │   ├── baseline.ipynb
│   │   ├── model1.ipynb
│   │   ├── model2.ipynb
│   │   ├── model3.ipynb
│   │   ├── cv.ipynb
│   │   ├── cv2.ipynb
│   │   ├── cv3.ipynb
│   │   └── testing.ipynb
│   │
│   ├── Submission/
│   │   ├── __init__.py
│   │   ├── submit.py
│   │   ├── submit.ipynb
│   │   └── submission.csv
│   │
│   ├── ReproducibilityCheck/
│   │   ├── __init__.py
│   │   ├── check_reproducibility.py
│   │   ├── reproducibilityCheck.ipynb
│   │   ├── generated_submission.csv
│   │   └── kaggle_submission.csv
│   │
│   ├── models/
│   │   ├── stack_bundle.joblib
│   │   └── submission.csv
│   │
│   ├── utils/
│   │   └── metric.py
│   │
│   └── __init__.py
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── entry_points.md
├── SETTINGS.json
├── directory_structure.txt
└── .python-version
```

## Important Files

### Final submission script

```text
src/Submission/submit.py
```

This is the main terminal-executable script. It loads the data, builds features, trains the model components, loads saved blending settings, blends predictions, and writes the final submission file.

### Original final notebook

```text
src/Submission/submit.ipynb
```

This is the original notebook workflow used for the final submission.

### Reproducibility check script

```text
src/ReproducibilityCheck/check_reproducibility.py
```

This script regenerates the final submission and compares it against the Kaggle-downloaded submission file.

### Saved blending settings

```text
src/models/stack_bundle.joblib
```

This file stores saved blending and post-processing settings used by the final submission workflow. It is not a fully trained model bundle. The final script retrains the model components during execution.

## Data

The expected data files are located in the `data/` folder:

```text
data/train.csv
data/val.csv
data/test.csv
data/sample_submission.csv
data/metaData.csv
```

The final submission workflow uses `train.csv`, `val.csv`, and `test.csv`. The training and validation files are combined during final model training.

No external datasets were used in the final solution. All features were derived from the competition-provided data.

## Environment

The project was developed with **Python 3.11**.

The repository includes both:

- `pyproject.toml` and `uv.lock` for `uv`
- `requirements.txt` for standard `pip` installation

## Running the Project with uv

This is the recommended way to reproduce the solution.

Run all commands from the top-level project directory.

### 1. Install dependencies

```bash
uv sync
```

### 2. Generate the final submission

```bash
uv run python -m src.Submission.submit
```

This writes the generated submission file to:

```text
src/Submission/submission.csv
```

Expected output includes the shape of the training and test feature matrices, model-training progress messages, the path to the saved submission file, a preview of the submission, and total runtime.

Example runtime from our tested environment:

```text
Total script runtime: 99.40 seconds
Total script runtime: 1 min 39.40 sec
```

### 3. Run the reproducibility check

```bash
uv run python -m src.ReproducibilityCheck.check_reproducibility
```

This regenerates the submission and saves it to:

```text
src/ReproducibilityCheck/generated_submission.csv
```

It then compares the regenerated file against:

```text
src/ReproducibilityCheck/kaggle_submission.csv
```

The reproducibility check verifies that the event IDs match and that the prediction values match within a small floating-point tolerance.

## Running the Project with pip and venv

Use this option if you are not using `uv`.

Run all commands from the top-level project directory.

### 1. Create a virtual environment

On Linux/macOS/WSL:

```bash
python3 -m venv .venv-pip
source .venv-pip/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv-pip
.venv-pip\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Generate the final submission

```bash
python -m src.Submission.submit
```

This writes the final submission file to:

```text
src/Submission/submission.csv
```

### 4. Run the reproducibility check

```bash
python -m src.ReproducibilityCheck.check_reproducibility
```

This writes the regenerated submission file to:

```text
src/ReproducibilityCheck/generated_submission.csv
```

and compares it against:

```text
src/ReproducibilityCheck/kaggle_submission.csv
```

### 5. Deactivate the environment

```bash
deactivate
```

## Expected Output Files

After running the final submission script:

```text
src/Submission/submission.csv
```

After running the reproducibility check:

```text
src/ReproducibilityCheck/generated_submission.csv
```

The Kaggle-downloaded submission used for comparison should be located at:

```text
src/ReproducibilityCheck/kaggle_submission.csv
```

## Reproducibility Notes

The final script retrains the model components every time it runs. This is expected behavior for this solution.

The saved file:

```text
src/models/stack_bundle.joblib
```

contains the saved blending and post-processing settings used by the final workflow.

The reproducibility check follows the same logic as the final submission workflow, saves the regenerated predictions to CSV, reloads them, and compares them against the Kaggle-downloaded submission file. This avoids confusion from tiny in-memory floating-point formatting differences.

## Model Details

The final solution includes:

- Horizon-specific modeling for 12h, 24h, 48h, and 72h predictions
- Censor-aware labels and inverse probability of censoring weighting
- Seed-bagged LightGBM, XGBoost, and CatBoost models
- XGBoost AFT survival modeling
- Cox-style survival modeling experiments
- Per-horizon blending
- Post-processing to ensure valid cumulative probabilities

The feature set includes distance features, distance-trend features, movement features, ETA features, fire size and growth features, directional alignment features, and temporal/observation-quality features.

## Project Notes

- Run scripts from the top-level project directory.
- Use `python -m ...` instead of running files by path when using the package-style commands.
- The folder names are case-sensitive on Linux/WSL.
- The final model uses only competition-provided data.
- The generated submission file follows the Kaggle sample submission format.

## References

- Kaggle competition overview: https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/overview
- Kaggle competition data page: https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/data
- LightGBM documentation: https://lightgbm.readthedocs.io
- XGBoost documentation: https://xgboost.readthedocs.io
- CatBoost documentation: https://catboost.ai
- scikit-learn documentation: https://scikit-learn.org
- pandas documentation: https://pandas.pydata.org
- NumPy documentation: https://numpy.org
- lifelines documentation: https://lifelines.readthedocs.io
- scikit-survival documentation: https://scikit-survival.readthedocs.io
