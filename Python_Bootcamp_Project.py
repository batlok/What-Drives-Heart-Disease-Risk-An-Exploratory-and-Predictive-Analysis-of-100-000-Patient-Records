# Load dataset
import pandas as pd
df = pd.read_csv("heart_disease_prediction_2026.csv")
print(f"Shape: {df.shape}")
df.head(10)

# Data overview
df.info()
print(df.dtypes.value_counts())
df.describe()
cat_cols = df.select_dtypes(exclude="number").columns

for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())
    # Check for label problems: extra whitespace, or the same label written in different cases
for col in cat_cols:
    s = df[col]
    print(col, "| extra whitespace:", (s != s.str.strip()).sum(),
          "| unique labels:", s.nunique(),
          "| unique labels after lowercasing:", s.str.lower().nunique())
    # Check missing values and duplicates
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
# Check outlier
num = [
    "age", "bmi", "sleep_hours", "resting_bp", "cholesterol",
    "max_heart_rate", "st_depression", "avg_resting_hr", "hrv_score"
]
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
for ax, c in zip(axes.flatten(), num):
    ax.boxplot(df[c], vert=False)
    ax.set_title(c)
plt.tight_layout(); plt.show()
for col in num:
    s = df[col].value_counts().sort_index()
    print(f"{col:<16} | Min values: {s.head(3).to_dict()}   |    Max values: {s.tail(3).to_dict()}")
    # Create columns for age group and BMI
df["age_group"] = pd.cut(df["age"],
                         bins=[0, 39, 54, 64, 120],
                         labels=["<40", "40-54", "55-64", "65+"])

df["bmi_category"] = pd.cut(df["bmi"],
                            bins=[0, 18.5, 25, 30, 100],
                            right=False,
                            labels=["Underweight", "Normal", "Overweight", "Obese"])

df["bloodPressure_category"] = pd.cut(df["resting_bp"],
                           bins=[0, 120, 130, 140, 1000],
                           right=False,
                           labels=["Normal", "Elevated", "High (stage 1)", "High (stage 2)"])
# Print them out to check whether they were  created correctly
print(df["age_group"].value_counts())
print(df["bmi_category"].value_counts())
print(df["bloodPressure_category"].value_counts())
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "axes.facecolor": "#f8f8f8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "white",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
})

BLUE = "#2979D6"
RED = "#E34A4A"
num = ["age", "bmi", "sleep_hours", "resting_bp", "cholesterol",
       "max_heart_rate", "st_depression", "avg_resting_hr", "hrv_score"]

summary = df.groupby("heart_disease")[num].mean().round(2).T
summary.columns = ["No disease", "Disease"]
summary["Difference"] = (summary["Disease"] - summary["No disease"]).round(2)
print(summary)
num = ["age", "bmi", "sleep_hours", "resting_bp", "cholesterol",
       "max_heart_rate", "st_depression", "avg_resting_hr", "hrv_score"]

fig, axes = plt.subplots(3, 3, figsize=(16, 11))
axes = axes.flatten()

for ax, col in zip(axes, num):
    bins = np.histogram_bin_edges(df[col], bins=25)   # cùng một bộ cột cho cả 2 nhóm
    ax.hist(df[df["heart_disease"] == 0][col], bins=bins, alpha=0.65, color=BLUE, label="No Disease")
    ax.hist(df[df["heart_disease"] == 1][col], bins=bins, alpha=0.65, color=RED, label="Heart Disease")
    ax.set_title(col)
    ax.set_ylabel("Count")

axes[0].legend()
fig.suptitle("Continuous Variables by Outcome", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# List of variables to plot (8 original categorical + 3 columns created in 2.3)
cat_vars = ["sex", "region", "income_level", "smoking_status",
            "alcohol_intake", "physical_activity", "resting_ecg", "st_slope",
            "age_group", "bmi_category", "bloodPressure_category"]

# Order of the groups from low to high (sex and region have no natural order, so they are skipped)
order = {
    "income_level": ["Low", "Middle", "High"],
    "smoking_status": ["Never", "Former", "Current"],
    "alcohol_intake": ["Non-drinker", "Moderate", "Heavy"],
    "physical_activity": ["Sedentary", "Light", "Active", "Very Active"],
    "resting_ecg": ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"],
    "st_slope": ["Upsloping", "Flat", "Downsloping"],
}

overall = df["heart_disease"].mean() * 100         # Overall heart disease rate (46.1%)

fig, axes = plt.subplots(3, 4, figsize=(18, 11))   # Create a grid of 3 rows x 4 columns of charts
axes = axes.flatten()                              # Turn the grid into a simple list of 12 panels

for ax, col in zip(axes, cat_vars):                # One variable per panel
    rate = df.groupby(col, observed=True)["heart_disease"].mean() * 100   # Heart disease rate of each group
    if col in order:
        rate = rate.loc[order[col]]                # Reorder the groups from low to high
    labels = [str(x).replace("Left Ventricular Hypertrophy", "LVH") for x in rate.index]   # Shorten the long label

    bars = ax.bar(labels, rate.values, color=RED, width=0.5)    # Draw the bars
    ax.bar_label(bars, fmt="%.1f%%", fontweight="bold")         # Write the % value on top of each bar
    ax.axhline(overall, linestyle="--", color="gray")           # Dashed line = overall rate
    ax.set_title(col)                                           # Chart title = variable name
    ax.set_ylim(0, 100)                                         # Y-axis always from 0 to 100% to make sure it's consistent over 12 charts
    ax.set_ylabel("% with Heart Disease")
    ax.tick_params(axis="x", rotation=20)                       # Rotate the x labels so they do not overlap

axes[-1].axis("off")      # Hide the empty last panel (11 variables, 12 panels)
fig.suptitle("Heart Disease Rate by Category (dashed line = overall 46.1%)", fontsize=16, fontweight="bold")
plt.tight_layout()        # Prevent charts from overlapping
plt.show()

import seaborn as sns

# Correlation between all numeric columns (18 columns, including heart_disease)
corr = df.select_dtypes("number").corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, annot_kws={"size": 8})   # annot=True writes the number in each cell
plt.grid(False)                                      # remove the grid lines from the global style
plt.title("Correlation Matrix of Numeric Variables", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()

# Heart disease rate (%) for each combination of age group and smoking status
table = df.pivot_table(index="age_group", columns="smoking_status",
                       values="heart_disease", aggfunc="mean", observed=True) * 100
table = table[["Never", "Former", "Current"]].round(1)   # put the columns in order
print(table)

# Line chart: one line per smoking status
table.plot(kind="line", marker="o", figsize=(8, 5))
plt.ylabel("% with Heart Disease")
plt.title("Heart Disease Rate by Age Group and Smoking Status", fontweight="bold")
plt.ylim(0, 100)
plt.show()

# Group smoking status into 2 categories: Smoker and Non-smoker
df["is_smoker"] = df["smoking_status"].apply(lambda x: "Smoker" if x != "Never" else "Non-smoker")

# Calculate the heart disease rate (%)
h1_table = pd.crosstab(df["is_smoker"], df["heart_disease"], normalize="index") * 100
print(h1_table.round(2))

# Calculate the heart disease rate (%) grouped by age group and physical activity level
h2_table = df.groupby(["age_group", "physical_activity"], observed=False)["heart_disease"].mean().unstack() * 100
print(h2_table.round(2))

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
            "This is not a validated personal medical risk estimate. If you think you have a medical problem please see a doctor."
        )
