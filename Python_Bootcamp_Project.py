import pandas as pd
df = pd.read_csv("heart_disease_prediction_2026.csv")
print(df.shape)
print(df["heart_disease"].value_counts())
print(df["heart_disease"].value_counts(normalize=True).round(3))
import matplotlib.pyplot as plt
ax = df["heart_disease"].value_counts().sort_index().plot(kind="bar", color=["#4C9F70", "#D9534F"])
ax.set_xticklabels(["No disease (0)", "Disease (1)"], rotation=0)
ax.set_title("Distribution of the target variable")
ax.set_ylabel("Number of patients")
plt.show()
groups = {
    "Demographics": ["age","sex","region","income_level"],
    "Lifestyle": ["smoking_status","alcohol_intake","physical_activity",
                  "diet_quality","sleep_hours","stress_level"],
    "Medical history": ["diabetes","hypertension","family_history","previous_heart_event"],
    "Clinical measures": ["resting_bp","cholesterol","fasting_blood_sugar","bmi"],
    "Cardiac tests": ["max_heart_rate","resting_ecg","exercise_angina",
                      "st_depression","st_slope","avg_resting_hr","hrv_score"],
}
# Kiểm tra: tổng số cột trong các nhóm phải là 25, và không thiếu cột nào
all_cols = sum(groups.values(), [])
print(len(all_cols))                                   # 25
print(set(df.columns) - set(all_cols))                 # {'heart_disease'}

# Đếm biến phân loại (cột chữ)
print(df.select_dtypes(exclude="number").columns.tolist())   # 8 cột

# 2.2 Data quality
print(df.shape, df.duplicated().sum(), df.isna().sum().sum())
for c in df.select_dtypes(exclude="number").columns:
    print(c, sorted(df[c].unique()))

num = ["age","bmi","sleep_hours","resting_bp","cholesterol",
       "max_heart_rate","st_depression","avg_resting_hr","hrv_score"]
rows = []
for c in num:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    n = ((df[c] < q1 - 1.5*iqr) | (df[c] > q3 + 1.5*iqr)).sum()
    rows.append((c, df[c].min(), df[c].max(), n, round(100*n/len(df), 2)))
print(pd.DataFrame(rows, columns=["var","min","max","n_outliers","pct"]))

# 2.3 Feature engineering
df["age_group"] = pd.cut(df.age, [0,39,54,64,120],
                         labels=["<40","40-54","55-64","65+"])
df["bmi_category"] = pd.cut(df.bmi, [0,18.5,25,30,100], right=False,
                            labels=["Underweight","Normal","Overweight","Obese"])
df["bp_category"] = pd.cut(df.resting_bp, [0,120,130,140,1000], right=False,
                           labels=["Normal","Elevated","High (stage 1)","High (stage 2)"])
df["heart_rate_reserve"] = df.max_heart_rate - df.avg_resting_hr
df["hr_vs_predicted"] = df.max_heart_rate / (220 - df.age)
df["risk_score"] = (df.diabetes + df.hypertension + df.family_history
                    + df.previous_heart_event
                    + (df.smoking_status == "Current").astype(int))

for c in ["age_group","bmi_category","bp_category","risk_score"]:
    print(df.groupby(c, observed=True).heart_disease.agg(["size","mean"]).round(3))

    import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Split the data into training (80%) and test (20%) sets
train, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["heart_disease"]
)

# 2. Define and fit the logistic regression model
# Dependent variable: heart_disease
# Control variables: age, resting_bp
# Lifestyle variables: sleep, stress, diet, smoking, alcohol, activity
# C() indicates a categorical variable
formula = """
heart_disease ~ age + resting_bp
+ sleep_hours + stress_level + diet_quality
+ C(smoking_status)
+ C(alcohol_intake)
+ C(physical_activity)
"""

model = smf.logit(
    formula=formula,
    data=train,
    missing="raise"
).fit(maxiter=200)

print(model.summary())
print("Model converged:", model.mle_retvals["converged"])

# 3. Check multicollinearity using VIF
X = model.model.exog
variable_names = model.model.exog_names

vif_table = pd.DataFrame({
    "Variable": variable_names,
    "VIF": [
        variance_inflation_factor(X, i)
        for i in range(X.shape[1])
    ]
})

# Keep the intercept in the calculation, but exclude it from the report
vif_table = vif_table[vif_table["Variable"] != "Intercept"]

print("\nVIF results:")
print(vif_table.round(2))

# 4. Predict outcomes on the test set
probabilities = model.predict(test)
predictions = (probabilities >= 0.50).astype(int)

# 5. Evaluate model accuracy
y_test = test["heart_disease"]

print("\nModel accuracy:", round(
    accuracy_score(y_test, predictions), 3
))

# Rows: actual classes; columns: predicted classes (0, 1)
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# 6. Compare with always predicting the training majority class
majority_class = train["heart_disease"].mode().iloc[0]
baseline_accuracy = (y_test == majority_class).mean()

print("\nMajority-class baseline accuracy:", round(
    baseline_accuracy, 3
))

###### import pandas as pd

# Collect numerical inputs
  

age = float(input("Age: "))

if age < 29:
    age = 29
    print("Age adjusted to 29 for the model calculation.")
resting_bp = float(input("Resting blood pressure (mmHg): "))
sleep_hours = float(input("Sleep per night (hours): "))
stress_level = float(input("Stress level (1–10): "))
diet_quality = float(input("Diet quality (1–10): "))

# Show the exact category labels used in the training data
categories = [
    "smoking_status",
    "alcohol_intake",
    "physical_activity"
]

user_categories = {}

for column in categories:
    options = sorted(train[column].dropna().unique())
    print(f"\n{column} options: {', '.join(options)}")

    answer = input("Enter one option exactly as shown: ").strip()

    while answer not in options:
        answer = input("Invalid option. Please try again: ").strip()

    user_categories[column] = answer

# Create a one-row table with the same variable names as the model
user_data = pd.DataFrame({
    "age": [age],
    "resting_bp": [resting_bp],
    "sleep_hours": [sleep_hours],
    "stress_level": [stress_level],
    "diet_quality": [diet_quality],
    "smoking_status": [user_categories["smoking_status"]],
    "alcohol_intake": [user_categories["alcohol_intake"]],
    "physical_activity": [user_categories["physical_activity"]]
})

# Check basic input validity
if not (
    0 < age < 120
    and resting_bp > 0
    and 0 <= sleep_hours <= 24
    and 1 <= stress_level <= 10
    and 1 <= diet_quality <= 10
):
    print("Invalid numerical input. Please check your answers.")

else:
    # Avoid predictions outside the numerical ranges seen in training
    numerical_columns = [
        "age", "resting_bp", "sleep_hours",
        "stress_level", "diet_quality"
    ]

    outside_range = [
        column for column in numerical_columns
        if not (
            train[column].min()
            <= user_data[column].iloc[0]
            <= train[column].max()
        )
    ]

    if outside_range:
        print("Outside the training data range:", outside_range)
        print("No score displayed.")

    else:
        probability = model.predict(user_data).iloc[0]
        score = probability * 100

        print(f"\nEducational model score: {score:.1f}/100")
        print(
            "Based on synthetic data. "
            "This is not a validated personal medical risk estimate. If you think do you have a medical problem please see a doctor."
        )