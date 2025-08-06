# MLOps Workshop Project

---

## Overview

This is a project created for the MLOps workshop, demonstrating the end-to-end lifecycle of a machine learning model, from data collection to deployment.

---


## Problem Statement

The objective of this project is to build a machine learning model that can detect fraudulent credit card transactions. Fraudulent transactions are a significant problem for financial institutions, leading to substantial financial losses. We will develop a model to accurately classify a transaction as either legitimate or fraudulent.

---
## Objective

The primary objective is to build a binary classification model that can accurately distinguish between fraudulent and legitimate transactions. The model should have a high ability to identify fraudulent transactions while minimizing false positives to avoid inconveniencing legitimate customers.

---

## Model Inputs and Outputs

**Inputs:** The model will take a series of features related to a financial transaction as input. These may include:
* Transaction amount
* Time of the transaction
* Transaction history of the cardholder
* Location of the transaction

**Outputs:** The model will output a prediction for each transaction, which will be a binary classification:
* **0:** Legitimate transaction
* **1:** Fraudulent transaction

The model will also provide a probability score for its prediction.

## Evaluation Metrics

To evaluate the model's performance, we will primarily use the following metrics, which are well-suited for imbalanced datasets like this one:

* **Precision:** The ratio of correctly predicted fraudulent transactions to the total predicted fraudulent transactions.
* **Recall:** The ratio of correctly predicted fraudulent transactions to all actual fraudulent transactions.
* **F1-Score:** The harmonic mean of Precision and Recall, which provides a balanced view of the model's performance.

---

## Success Criteria

A successful model for this project will meet the following criteria:

* **F1-Score:** Achieve an F1-Score of at least 0.85 on the test dataset.
* **Performance:** The model must be able to make predictions within an acceptable latency to be suitable for real-time applications.

## Getting Started

### Prerequisites

* Python 3.10+
* Git

### Installation

1. Clone the repository:
    ```bash
    git clone [https://github.com/your-username/starter_mlops_project.git](https://github.com/your-username/starter_mlops_project.git)
    cd starter_mlops_project
    ```
2. Create and activate a virtual environment:
    ```bash
    # On Windows
    python -m venv .venv
    .venv\Scripts\Activate.ps1

    # On macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
