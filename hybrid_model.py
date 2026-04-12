import pandas as pd
import json
import joblib
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load trained ML model
model = joblib.load(
    os.path.join(BASE_DIR, "models", "rf_model.pkl")
)

# Load rules
with open(
    os.path.join(BASE_DIR, "rules", "rules.json"),
    "r"
) as f:
    rules_data = json.load(f)

high_rules = rules_data["high_risk_rules"]
low_rules = rules_data["low_risk_rules"]

# Selected columns used for reduced model
selected_cols = [
    "CheckingAccountStatus",
    "CreditHistory",
    "SavingsAccount",
    "Duration",
    "CreditAmount",
    "Age"
]


def match_rule(sample, rule_conditions):
    for feature, value in rule_conditions.items():
        if sample[feature] != value:
            return False
    return True


def hybrid_predict(sample):
    
    # Check High Risk rules first
    for rule in high_rules:
        if match_rule(sample, rule["conditions"]):
            return "High Risk (Rule-Based)", rule["confidence"]
    
    # Check Low Risk rules
    for rule in low_rules:
        if match_rule(sample, rule["conditions"]):
            return "Low Risk (Rule-Based)", rule["confidence"]
    
    # Otherwise use ML
    sample_df = pd.DataFrame([sample])
    sample_encoded = pd.get_dummies(sample_df)
    
    # Align columns with training data
    sample_encoded = sample_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )
    
    prediction = model.predict(sample_encoded)[0]
    
    if prediction == 0:
        return "High Risk (ML-Based)", None
    else:
        return "Low Risk (ML-Based)", None


# Example test
if __name__ == "__main__":
    
    test_sample = {
        "CheckingAccountStatus": 4,
        "CreditHistory": "Existing credits paid back",
        "SavingsAccount": "< 100 DM",
        "Duration": "Short",
        "CreditAmount": "Low",
        "Age": "Young"
    }
    
    result = hybrid_predict(test_sample)
    print("Prediction:", result)
