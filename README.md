# Sustainable Composite Beam AI

Research and decision-support software for dataset-driven structural performance prediction, SHAP explainability, and sustainable material optimization. Numerical values come only from trained regressors, SHAP, dataset constraints, and optimization algorithms. Hugging Face is optional and is limited to parsing and explaining verified results.

The original Streamlit review application remains available. The complete dynamic application uses a FastAPI backend and Vite/React frontend, with CSV, XLSX, JSON, and Joblib files instead of a database.

## Capabilities

- CSV/XLSX upload, preview, statistics, missing values, duplicates, IQR outliers, suspicious ranges, correlations, and strong-correlation warnings.
- Dynamic column roles, labels, units, categories, continuous/discrete metadata, optimization flags, and sustainability objectives.
- Leakage-safe sklearn pipelines for numerical and categorical inputs.
- Single- and multi-target regression with Linear, Ridge, Random Forest, Extra Trees, Gradient Boosting, MLP, XGBoost, LightGBM when installed, and CatBoost.
- Holdout metrics, cross-validation, best-model selection, timestamped model versions, and active model metadata.
- Dynamic prediction forms, experimental-domain classification, prediction history, and explicit uncertainty limitations.
- Per-target local and global SHAP values with deterministic engineering-language fallback.
- Bounded random/grid, genetic, and Pareto/NSGA-II optimization with fixed, categorical, stepped, and discrete variables.
- Optional Hugging Face API or local Transformers explanations; numerical services continue when HF is unavailable.
- JSON/CSV exports for predictions, optimization solutions, model metrics, and SHAP importance.

## Safety

This system is a research and decision-support tool based on experimental data and machine-learning predictions. Final structural design must be verified using applicable design codes, engineering calculations, and qualified professional review.

SHAP describes how the trained model used each feature. It does not prove physical causality. Optimization never expands beyond the observed experimental feature ranges.

## Architecture

```text
CSV/XLSX -> validation -> saved schema -> preprocessing pipeline
         -> model comparison -> active model
         -> prediction -> domain guard -> SHAP
         -> grid / GA / NSGA-II optimization
         -> optional HF explanation -> React dashboard -> JSON/CSV export
```

- `backend/app/api/routes/`: FastAPI endpoints
- `backend/app/ml/`: loading, validation, training, prediction, and registry
- `backend/app/explainability/`: local/global SHAP
- `backend/app/optimization/`: constrained search and Pareto filtering
- `backend/app/hf/`: optional parsing and explanation with fallback
- `backend/config/`: file-based dataset/model state
- `backend/artifacts/`: versioned models, preprocessors, reports, and exports
- `frontend/src/pages/`: routed engineering workflow
- `app.py`: preserved Streamlit review application

## Requirements

- Windows
- Python 3.12
- Node.js 20 or newer

## Backend Setup

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

API documentation is available at `http://localhost:8000/docs`.

## Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Copy `frontend/.env.example` to `frontend/.env` only when the API URL needs to change.

After dependencies are installed, `run_project.bat` starts both services in separate Windows command windows.

## Streamlit Review App

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit app uses the established artifacts under the root `models/` folder and is isolated from dynamic backend artifacts.

## Workflow

1. Open **Dataset** and upload a CSV or XLSX workbook.
2. Assign every column as input, target, identifier, or ignored; set units and optimization metadata.
3. Review quality findings and save the schema.
4. Open **Model Training**, choose split/CV settings, and train candidate models.
5. Review the selected model and metrics.
6. Open **Prediction**, enter dataset-derived inputs, and inspect outputs, domain status, SHAP, and history.
7. Open **Optimization**, set structural targets and select baseline, GA, or NSGA-II.
8. Inspect alternatives and the Pareto front in **Results**, then download reports.

The included `backend/data/sample/synthetic_beam_demo.csv` is labelled **SYNTHETIC DEMONSTRATION DATA - NOT FOR ENGINEERING USE**. A real dataset can replace it without source changes; remap columns in the Dataset page.

## Hugging Face

Configure `backend/.env` from `backend/.env.example`:

```dotenv
HF_PROVIDER=disabled
HF_MODEL_ID=
HF_TOKEN=
```

Use `HF_PROVIDER=api` for the Inference API or `HF_PROVIDER=local` for Transformers. If configuration, network access, output parsing, or inference fails, deterministic explanations and parsing remain available. Never commit a real token.

## Main API

- `GET /api/health`
- `POST /api/dataset/upload`
- `GET /api/dataset/info`
- `POST /api/dataset/schema`
- `GET /api/dataset/quality`
- `GET /api/dataset/ranges`
- `POST /api/training/train`
- `GET /api/training/status`
- `GET /api/training/results`
- `GET /api/models/best`
- `POST /api/predict`
- `GET|DELETE /api/prediction/history`
- `POST /api/explain/local`
- `GET /api/explain/global`
- `POST /api/optimization/parse-request`
- `POST /api/optimization/run`
- `GET /api/optimization/config`
- `GET /api/reports/model-comparison`
- `GET /api/reports/prediction`
- `GET /api/reports/optimization`
- `GET /api/reports/feature-importance`

JSON endpoints use `{ "success", "message", "data" }`; errors use `{ "success": false, "message", "details" }`.

## Tests

```powershell
cd backend
python -m pytest -q

cd ..\frontend
npm test
npm run build
```

## Limitations

- Predictions are empirical ML estimates and are unreliable outside representative data coverage.
- The displayed domain-similarity label is descriptive, not a calibrated confidence interval.
- HF text is explanatory only and has no numerical authority.
- Cost or carbon objectives are available only when the uploaded dataset and schema explicitly provide those fields.
- Large local Transformer models may require substantial memory; HF can remain disabled.
