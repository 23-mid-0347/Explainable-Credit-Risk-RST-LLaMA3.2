import streamlit as st
import pandas as pd
import json
import joblib
import ollama 
import os

# -----------------------------
# 1. Load Model & Rules
# -----------------------------
@st.cache_resource
def load_assets():
    try:
        # Update these paths to match your local machine
        model_path = r"D:\Coding\6th sem\Soft Computing\Credit Risk Assessment\models\rf_model.pkl"
        rules_path = r"D:\Coding\6th sem\Soft Computing\Credit Risk Assessment\rules\rules.json"
        
        model = joblib.load(model_path)
        with open(rules_path, "r") as f:
            rules_data = json.load(f)
        return model, rules_data
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

model, rules_data = load_assets()

# -----------------------------
# 2. Hybrid Logic Functions
# -----------------------------
def match_rule(sample, rule_conditions):
    for feature, value in rule_conditions.items():
        if sample[feature] != value:
            return False
    return True

def hybrid_predict(sample):
    high_rules = rules_data.get("high_risk_rules", [])
    low_rules = rules_data.get("low_risk_rules", [])

    # 1. Check Symbolic Rules (High Risk)
    for rule in high_rules:
        if match_rule(sample, rule["conditions"]):
            return "High Risk", "Rule-Based", rule
    
    # 2. Check Symbolic Rules (Low Risk)
    for rule in low_rules:
        if match_rule(sample, rule["conditions"]):
            return "Low Risk", "Rule-Based", rule
    
    # 3. Machine Learning Fallback (Random Forest)
    sample_df = pd.DataFrame([sample])
    sample_encoded = pd.get_dummies(sample_df)
    sample_encoded = sample_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )
    
    prediction_idx = model.predict(sample_encoded)[0]
    prediction = "High Risk" if prediction_idx == 0 else "Low Risk"
    return prediction, "ML-Based", None

# -----------------------------
# 3. Python Reasoning (The "What")
# -----------------------------
def generate_ground_truth(sample, prediction, method, rule):
    """Generates the factual core logic to prevent LLM hallucinations."""
    if method == "Rule-Based":
        return (f"The application was assessed against specific safety criteria. "
                f"While the history is '{sample['CreditHistory']}', the current combination of "
                f"'{sample['SavingsAccount']}' savings and a '{sample['Duration']}' term "
                f"suggests the financial safety net is still being established.")
    else:
        return (f"The system analyzed the overall balance of the profile. It determined that "
                f"the relationship between the '{sample['CreditAmount']}' amount requested, "
                f"the '{sample['Age']}' age group, and current '{sample['SavingsAccount']}' reserves "
                f"aligns with a {prediction} profile based on historical patterns.")

# -----------------------------
# 4. Llama 3.2 Explanation (The "Why")
# -----------------------------
def generate_explanation(reasoning):
    system_prompt = (
        "You are a professional Senior Financial Consultant speaking directly to a client. "
        "Explain their credit assessment results in a natural, supportive way. "
        "\n\nSTRICT FORMATTING RULES:"
        "\n- Do NOT use headers like 'Paragraph 1', 'Part', or 'Heading'."
        "\n- Write exactly two short, cohesive paragraphs."
        "\n- Paragraph 1: Explain that while their history is reliable, low savings create a 'liquidity gap' making them vulnerable to emergencies."
        "\n- Paragraph 2: Give encouraging advice on building a cash buffer to strengthen their future profile."
        "\n- Do NOT mention 'technical logic', 'prediction', or 'rules'."
        "\n- Be warm and professional."
    )
    
    try:
        response = ollama.chat(
            model='llama3.2', 
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Context: {reasoning}"},
            ],
            options={
                'temperature': 0.3, 
                'num_predict': 300,
                'stop': ["Paragraph", "Part:", "Logic:", "Result:"]
            }
        )
        
        explanation = response['message']['content'].strip()
        
        # Cleanup any accidental leading prefixes
        prefixes = ["Here is the explanation:", "Based on the assessment:", "Consultant's Insight:"]
        for p in prefixes:
            if explanation.startswith(p):
                explanation = explanation[len(p):].strip()
        return explanation

    except Exception as e:
        return f"Assessment Summary: {reasoning}"

# -----------------------------
# 5. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Credit Insight AI", page_icon="🏦", layout="wide")
st.title("🏦 Hybrid Explainable Credit Risk System")
st.markdown("---")

# Layout with Columns
col_in, col_out = st.columns([1, 1.2], gap="large")

with col_in:
    with st.container(border=True):
        st.subheader("Applicant Profile")
        checking = st.selectbox("Account Status", [1, 2, 3, 4])
        history = st.selectbox("Credit History", [
            "All credits paid back", "Critical account", "Delay in paying", 
            "Existing credits paid back", "No credit history"
        ])
        savings = st.selectbox("Savings Balance", [
            "< 100 DM", "100 <= savings < 500", "500 <= savings < 1000", 
            ">= 1000 DM", "Unknown / None"
        ])
        
        st.subheader("Loan Requirements")
        duration = st.selectbox("Loan Duration", ["Short", "Medium", "Long"])
        amount = st.selectbox("Credit Amount", ["Low", "Medium", "High"])
        age = st.selectbox("Age Group", ["Young", "Middle", "Old"])
        
        submit = st.button("Run Assessment", use_container_width=True, type="primary")

with col_out:
    if submit:
        sample = {
            "CheckingAccountStatus": checking,
            "CreditHistory": history,
            "SavingsAccount": savings,
            "Duration": duration,
            "CreditAmount": amount,
            "Age": age
        }

        # Step 1: Logic
        prediction, method, rule = hybrid_predict(sample)
        base_logic = generate_ground_truth(sample, prediction, method, rule)

        # Step 2: AI Explanation
        with st.spinner("Llama 3.2 is analyzing the liquidity gap..."):
            explanation = generate_explanation(base_logic)

        # Step 3: UI Display
        st.subheader("Assessment Result")
        color = "red" if prediction == "High Risk" else "green"
        st.markdown(f"### Result: :{color}[{prediction}]")
        st.caption(f"Engine: {method}")
        
        st.markdown("---")
        st.subheader("Guidance & Insights")
        st.success(explanation)
        
        # New: Feature Importance Visualization
        st.markdown("---")
        st.subheader("Decision Influence")
        
        # Get importance from the ML model
        importances = model.feature_importances_
        feature_names = model.feature_names_in_
        
        # Create a DataFrame for the chart
        feat_importances = pd.Series(importances, index=feature_names)
        
        # Filter for the most important features to keep the chart clean
        top_features = feat_importances.nlargest(5)
        
        st.bar_chart(top_features)
        st.caption("This chart shows which factors most heavily influenced the AI's decision.")

        with st.expander("View Technical Reasoning (Internal Logic)"):
            st.code(base_logic)
    else:
        st.info("Fill out the profile on the left and click 'Run Assessment' to see the AI analysis.")
    
    