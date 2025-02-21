# MLOps-Salary-project

![CI/CD/CT Pipeline](https://img.shields.io/badge/CI/CD/CT-Automated-blue.svg)

## 🚀 Overview
This project implements a **CI/CD/CT (Continuous Integration, Continuous Deployment, Continuous Training)** pipeline using **Azure Web App, FastAPI, MLflow, DVC, and Azure Blob Storage**. It automates data versioning, model training, experiment tracking, and model deployment

## 📌 Features
- **CI/CT/CD:**
  - **CI (Continuous Integration)**: Automates code testing and validation on each commit
  - **CT (Continuous Training)**: Triggers model retraining when new data is available
  - **CD (Continuous Deployment)**: Deploys the latest trained model to production automatically

- **Storage:**
  - **Azure Blob Storage:**
    - `Raw/` → Stores incoming raw data
    - `Data Versions/` → Logs and stores training, testing, and manipulated data using **DVC**
    - `Model Artifacts/` → Stores model artifacts (`model.pkl`, `best_params.json`, etc) via **MLflow**
  - **Azure SQL Server & DB:** Stores **model metadata** (metrics, params, name, duration, timestamp)

- **Model Tracking & Versioning:**
  - **DVC:** Manages data versioning
  - **MLflow:** Tracks models, logs artifacts & metadata
  - **MLflow UI:** Deployed separately on Azure Web App

- **Automation & Security:**
  - **Docker:** Containerizes final serving block of code
  - **GitHub Actions:** Automates workflows
  - **Azure Key Vault & GitHub Secrets:** Secures credentials
  - **Azure Functions & Event Triggers:** Detects raw data changes to trigger pipeline


## 📁 Project Structure




## 🛠️ Technologies Used

### 🚀 **Cloud & Infrastructure**
- **Azure Web App** - Hosts FastAPI services for model deployment.
- **Azure Blob Storage** - Stores raw data, processed datasets, and model artifacts.
- **Azure SQL Server & DB** - Stores model metadata (metrics, parameters, training history).
- **Azure Functions & Event Triggers** - Automates training when new data is added.

### 🏗 **Model Development & Tracking**
- **Python** - Core programming language for data processing, training, and inference.
- **FastAPI** - Serves the trained model via API.
- **MLflow** - Logs models, artifacts, and experiment metadata.
- **DVC (Data Version Control)** - Manages dataset versions.

### 🔁 **CI/CD & Automation**
- **GitHub Actions** - Automates CI/CD pipelines.
- **Docker** - Containerizes ML services (FastAPI, MLflow UI).
- **Azure Key Vault** - Manages credentials securely.

### 📊 **Monitoring & Logging**
- **MLflow UI** - Deployed on Azure Web App to track model experiments.
- **Azure Monitor** - Logs application events and metrics.

### 📂 **Configuration**
- **YAML** - Stores constants, parameters, and pipeline configurations.
- **GitHub Secrets** - Stores sensitive credentials securely.

## 🔁 Workflow Diagram
![](https://github.com/Senan25/MLOps-Salary-project/blob/main/Untitled%20Diagram.drawio.svg)
