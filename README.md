# What-Drives-Heart-Disease-Risk-An-Exploratory-and-Predictive-Analysis-of-100-000-Patient-Records

# What Drives Heart Disease Risk?

An exploratory and predictive analysis of 100,000 synthetic patient records, developed as a Python Bootcamp project at KEDGE Business School.

The project asks two questions. How does the proportion of records labeled with heart disease vary across age, smoking and physical activity groups? How accurately can a logistic regression classify heart disease status using age, resting blood pressure and lifestyle variables?

## Dataset

The repository includes the [dataset](heart_disease_prediction_2026.csv) and the [Python analysis script](Python_Bootcamp_Project.py). The CSV contains **100,000 rows and 26 columns**, with no missing values or duplicate rows. The binary outcome, `heart_disease`, equals `1` for 46,082 records and `0` for 53,918 records.

## Exploratory findings

| Age group | Records labeled with heart disease |
| --- | ---: |
| Under 40 | 12.5% |
| 40–54 | 33.4% |
| 55–64 | 57.9% |
| 65 and older | 79.3% |

Across all ages, **57.4%** of records in the current-smoker group had a positive label, compared with **47.1%** among former smokers and **43.1%** among those who never smoked. The corresponding proportions ranged from **53.1%** in the sedentary group to **38.4%** in the very active group.

These percentages describe unadjusted associations in the synthetic dataset. Because the outcome varies substantially with age, the script also compares smoking and physical activity categories **within each age group**. Histograms, box plots and a correlation heatmap provide additional views of the underlying variables.

## Predictive model

The script fits a binary logistic regression with eight predictors: age, resting blood pressure, sleep duration, stress level, diet quality, smoking status, alcohol intake and physical activity. It reports model convergence and variance inflation factors to check for multicollinearity.

A stratified 80/20 split preserves the outcome proportions in the training and test sets (`random_state=42`). At a classification threshold of `0.50`, the model correctly classified **14,595 of 20,000** held-out records.

| Method | Test accuracy |
| --- | ---: |
| Logistic regression | 72.98% |
| Majority-class baseline | 53.92% |

**Confusion matrix on the test set**

| Observed outcome | Predicted no disease | Predicted disease |
| --- | ---: | ---: |
| No disease | 8,204 | 2,580 |
| Disease | 2,825 | 6,391 |

The model identified 6,391 of the 9,216 positive cases in the test set. Its score estimates the probability of a positive label in this teaching dataset. Clinical use would require validation on real patient data.

## Run the project

Download or clone the repository and open a terminal in its root folder. Keep the CSV beside the Python script. Use a local Python 3 installation with Tkinter and a graphical display.

```bash
python -m pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
python Python_Bootcamp_Project.py
```

The script runs in three stages.

1. **Explore the data.** The terminal prints data-quality checks and summary tables. Close each plot window to advance through the analysis.
2. **Evaluate the model.** The terminal prints the regression summary, test accuracy, confusion matrix and baseline accuracy. It then prompts for an example profile and displays an educational model score for accepted inputs.
3. **Explore an age group.** The Tkinter app displays record counts and disease-label percentages by smoking status and physical activity for a selected age group. It summarizes the two factors separately. The script includes two versions of the app, so closing the first window opens the second.
