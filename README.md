# TelemetryPCA: Multivariate Exploration of Spacecraft Telemetry Data

This repository contains the implementation of the analyses defined within the EUMETSAT telemetry data exploration workflow.

## Repository Structure

### Task 2: Single Payload Analysis
Python script implementing multivariate analysis for temperature telemetry data from a single payload. The workflow includes:

- Data preprocessing
- Principal Component Analysis (PCA)
- Visualization of PCA outputs and derived results
- Computation of seasonal variability and long term trends
- Exploratory analysis for anomaly identification

### Task 3: Multi Mission Analysis
Python script implementing multivariate analysis across multiple satellite missions. The workflow includes:

- Data preprocessing for heterogeneous telemetry datasets
- Principal Component Analysis (PCA)
- Visualization of multivariate outputs
- Computation of seasonal variability and long term trends
- Cross mission exploratory analysis

## Input Data

Input datasets are organized according to the telemetry data streams defined within **Task 1**. Preprocessing implemented in both scripts follows the methodology and workflow described in the corresponding **Task 1 report**.

## Documentation

Interpretation of results, methodological comments, conclusions, and explanations supporting the implemented functions are not embedded within the scripts and are documented in the respective **Task 2** and **Task 3 reports**.
