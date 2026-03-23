# Explainable Credit Risk Assessment using Rough Set Theory and LLaMA 3.2

## Overview

This project presents a hybrid Explainable AI (XAI) system for credit risk assessment by integrating:

Rough Set Theory (rule-based reasoning)
Machine Learning models (prediction)
Meta LLaMA 3.2 (natural language explanations)

The system not only predicts whether a loan applicant is high or low risk but also explains the decision in both logical and human-readable form.

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

## Tech Stack

* Python
* Pandas
* Scikit-learn
* HuggingFace Transformers / Ollama
* LLaMA 3.2

---

## Future Work

* Streamlit Web Interface
* Real-time prediction system
* Model optimization
* Deployment

---

## Author

Shyam Prasad
