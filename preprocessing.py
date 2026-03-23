import pandas as pd

df = pd.read_csv("D:\\Coding\\6th sem\\Soft Computing\\Credit Risk Assessment\\data\\german_credit_data.csv")

# Rename columns
df.rename(columns={
    "laufkont": "CheckingAccountStatus",
    "laufzeit": "Duration",
    "moral": "CreditHistory",
    "verw": "Purpose",
    "hoehe": "CreditAmount",
    "sparkont": "SavingsAccount",
    "beszeit": "EmploymentDuration",
    "rate": "InstallmentRate",
    "famges": "PersonalStatus",
    "buerge": "Guarantor",
    "wohnzeit": "ResidenceDuration",
    "verm": "Property",
    "alter": "Age",
    "weitkred": "OtherInstallmentPlans",
    "wohn": "Housing",
    "bishkred": "ExistingCredits",
    "beruf": "Job",
    "pers": "Dependents",
    "telef": "Telephone",
    "gastarb": "ForeignWorker",
    "kredit": "Risk"
}, inplace=True)

print(df.columns)

print(df["Risk"].unique())
print(df["Risk"].value_counts())

# Correct mapping
df["Risk"] = df["Risk"].map({
    1: "Low Risk",
    0: "High Risk"
})

print(df["Risk"].value_counts())

df.to_csv("D:\\Coding\\6th sem\\Soft Computing\\Credit Risk Assessment\\data\\Cleaned_german_data.csv", index=False)
