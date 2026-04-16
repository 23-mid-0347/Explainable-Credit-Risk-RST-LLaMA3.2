# Explainable Credit Risk Assessment using Rough Set Theory and LLaMA 3.2

## Overview

This project presents a hybrid Explainable AI (XAI) system for credit risk assessment by integrating:

Rough Set Theory (rule-based reasoning)
Machine Learning models (prediction)
Meta LLaMA 3.2 (natural language explanations)

The system not only predicts whether a loan applicant is high or low risk but also explains the decision in both logical and human-readable form.

---
## Tech Stack

Language: Python 3.10+

Machine Learning: Scikit-learn, Pandas, NumPy

LLM Integration: Ollama / HuggingFace Transformers (Meta LLaMA 3.2)

Web Framework: Flask (for API access)
---
## 📂 Project Structure

.
├── data/                   # Raw and processed datasets (German Credit Dataset)
├── data_preprocessing/     # Scripts for cleaning and discretization
├── models/                 # Saved ML model weights and logic
├── rules/                  # Generated IF-THEN rules from RST
├── preprocessing.py        # Core data transformation logic
├── ml_models.py            # Training and evaluation script
├── hybrid_model.py         # Integration logic (RST + LLaMA)
├── app.py                  # Flask application for real-time inference
└── README.md
---
## Key Features

* Complete data preprocessing pipeline
* Categorical decoding of financial attributes
* Quantile-based discretization
* Rough Set-based feature reduction (Reduct)
* Rule extraction (IF–THEN rules)
* ML baseline comparison (SVM, Random Forest)
* Natural language explanations using LLaMA 3.2

---
## 🚀 Getting Started
1. Clone the Repository
2. Set Up Environment
It is recommended to use a virtual environment:
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate

    pip install -r requirements.txt
3. Set Up LLaMA 3.2 (via Ollama)
    This project uses LLaMA 3.2 for generating human-readable explanations.

    Install Ollama.

    Pull the LLaMA 3.2 model: ollama run llama3.2
---
### ⚙️ How to Run
# Step 1: Data Preprocessing & Rule Extraction
Run the preprocessing script to clean the German Credit dataset and discretize continuous variables.

Bash
python preprocessing.py

# Step 2: Train Baseline Models
Train the SVM and Random Forest models to compare performance against the RST rules.

Bash
python ml_models.py

# Step 3: Run the Flask Application
Start the server to get predictions and natural language explanations.

Bash
python app.py
---
## Dataset

* German Credit Dataset (UCI)
* 1000 applicants
* 20 attributes + 1 decision attribute

---

## Workflow

1. Data Cleaning
2. Categorical Decoding
3. Discretization
4. Rough Set Analysis
5. Feature Reduction (Reduct)
6. Rule Extraction
7. ML Model Training
8. LLaMA 3.2 Explanation Generation

---

## Example Rule

IF
CreditHistory = Critical account
AND SavingsAccount = < 100 DM
AND DurationCategory = Long

THEN High Risk

---

## Example Explanation (LLaMA 3.2)

The applicant is classified as high risk due to a poor credit history and limited savings, along with a long loan duration, which increases the likelihood of repayment issues.

---

## Author

Shyam Prasad
Pooja s
Arul Amuthan
GitHub: @23-mid-0347
