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
