# Colorectal Cancer Survival Prediction - MLOps Pipeline

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=flat&logo=mlflow&logoColor=blue)
![Optuna](https://img.shields.io/badge/Optuna-blue.svg)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![Kubeflow](https://img.shields.io/badge/kubeflow-%234285F4.svg?style=flat&logo=kubeflow&logoColor=white)
![UV](https://img.shields.io/badge/uv-astral-blue.svg)
![Ruff](https://img.shields.io/badge/Ruff-checked-green.svg)

## 🎯 Objectives

### Technical Objective
The primary technical goal of this project is to build a classification pipeline to predict **Colorectal Cancer Survival** ("Yes" or "No"). Using a dataset of clinical features, the pipeline performs automated feature selection, hyperparameter optimization, and model training to deliver accurate survival predictions.

### MLOps Objective (The "Real" Goal)
Beyond the prediction itself, this project serves as a **MLOps Portfolio Piece**. It is designed to demonstrate professional engineering standards in a machine learning context. Key MLOps practices showcased include:
- **Reproducibility:** Using `uv` for lightning-fast, deterministic dependency management.
- **Experiment Tracking:** Logging parameters, metrics, and models using **MLflow**.
- **Automated Optimization:** Hyperparameter tuning with **Optuna**.
- **Containerization:** A production-ready **Docker** environment.
- **Orchestration:** Cloud-native pipeline definition using **Kubeflow Pipelines**.
- **Code Quality:** Automated linting and formatting with **Ruff** and **Pre-commit** hooks.
- **Adaptive Serving:** A Flask web application that dynamically adapts to different feature sets selected during the training phase.

---

## 🏗️ Architecture

1.  **Data Processing:** Cleans data, performs label encoding for categorical variables (including the target), and selects the top 5 features using Chi-Square.
2.  **Training:** Optimizes a `GradientBoostingClassifier` using Optuna and logs the entire process to MLflow.
3.  **Serving:** A Flask-based UI that loads the saved model and metadata (scalers, encoders) to provide real-time predictions.
4.  **Pipeline:** The entire flow is defined as a Kubeflow Pipeline for scalable execution.

---

## 🚀 Getting Started

### Prerequisites
- [uv](https://docs.astral.sh/uv/) (Recommended for speed and reliability)
- [Docker](https://www.docker.com/)

### Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mikemayuare/colorectal-cancer-prediction.git
   cd colorectal-cancer-prediction
   ```

2. **Sync dependencies:**
   ```bash
   uv sync
   ```

3. **Run the pipeline manually:**
   ```bash
   # Process data
   uv run -m src.processing
   
   # Train model
   uv run -m src.training
   ```

4. **Start the Flask App:**
   ```bash
   uv run main.py
   ```
   Visit `http://localhost:5000` in your browser.

### Docker Usage

**Build and Run:**
```bash
docker build -t colorectal-prediction .
docker run -p 5000:5000 colorectal-prediction
```

---

## ☁️ Kubeflow Integration

The project includes a pipeline definition in `kubeflow_pipeline/pipeline.py`. To compile the pipeline into a `pipeline.yaml` file for upload to a Kubeflow cluster:

```bash
uv run -m kubeflow_pipeline.pipeline
```

---

## 📊 Experiment Tracking

To view your training runs and model metrics, launch the MLflow UI:

```bash
uv run mlflow ui
```

---

## 🛠️ Project Structure

```text
├── artifacts/           # Data and trained model binaries
├── config/              # Path configurations
├── kubeflow_pipeline/   # KFP pipeline definitions
├── src/                 # Core logic (processing, training, logging)
├── static/ & templates/ # Flask UI assets
├── main.py              # Application entry point
├── Dockerfile           # Container definition
└── pyproject.toml       # uv/python configuration
```

