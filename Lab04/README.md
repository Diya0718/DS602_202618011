# 🏥 Medical Insurance Cost Analytics

An interactive Streamlit application for exploring medical insurance data, performing statistical hypothesis testing, building a multiple linear regression model, and predicting medical insurance charges.

## 🚀 Live Application

**Streamlit App:**  
https://medical-insurance-cost-analytics.streamlit.app/

**GitHub Repository:**  
https://github.com/Diya0718/202618011_DS602/tree/main/Lab04

---

## 📌 Project Overview

Medical insurance charges can vary considerably depending on factors such as age, BMI, smoking status, number of children, sex, and region.

This project performs:

- Exploratory Data Analysis (EDA)
- Descriptive statistical analysis
- Hypothesis testing
- One-Way ANOVA
- Multiple Linear Regression using OLS
- Regression diagnostics
- Heteroscedasticity testing
- VIF analysis
- HC3 heteroscedasticity-robust regression
- Interactive medical charge prediction using Streamlit

The application provides an interactive interface for exploring the dataset, conducting statistical tests, and generating predictions.

---

# 📊 Dataset Summary

The project uses the **Medical Insurance Costs (`insurance.csv`)** dataset.

- **Number of observations:** 1,338
- **Number of variables:** 7
- **Missing values:** 0
- **Target variable:** `charges`

### Dataset Variables

| Variable | Description | Type |
|---|---|---|
| `age` | Age of the individual | Numerical |
| `sex` | Sex of the individual | Categorical |
| `bmi` | Body Mass Index | Numerical |
| `children` | Number of dependent children | Numerical |
| `smoker` | Smoking status | Categorical |
| `region` | Residential region | Categorical |
| `charges` | Medical insurance charges | Numerical |

---

# 📈 Exploratory Data Analysis

The numerical variables analyzed were:

- Age
- BMI
- Number of children
- Medical charges

The analysis included:

- Mean
- Median
- Standard deviation
- Interquartile range (IQR)
- Skewness
- Kurtosis
- Histograms and KDE plots
- Scatter plots
- Correlation analysis

## Descriptive Statistics

| Variable | Mean | Median | Std Dev | IQR | Skewness | Kurtosis |
|---|---:|---:|---:|---:|---:|---:|
| Age | 39.2070 | 39.0000 | 14.0500 | 24.0000 | 0.0557 | -1.2451 |
| BMI | 30.6634 | 30.4000 | 6.0982 | 8.3975 | 0.2840 | -0.0507 |
| Children | 1.0949 | 1.0000 | 1.2055 | 2.0000 | 0.9384 | 0.2025 |
| Charges | 13270.4223 | 9382.0330 | 12110.0112 | 11899.6254 | 1.5159 | 1.6063 |

### Interpretation

Age and BMI are relatively close to symmetric distributions, while the number of children shows moderate positive skewness.

Medical charges show substantial positive skewness. The mean charge ($13,270.42) is considerably higher than the median ($9,382.03), indicating the presence of relatively high-cost observations.

---

# 🧪 Hypothesis Testing

## Hypothesis Test 1 — Smokers vs Non-Smokers

The first hypothesis test compares medical charges between smokers and non-smokers.

### Group Summary

| Group | Count | Mean Charges |
|---|---:|---:|
| Smokers | 274 | $32,050.23 |
| Non-smokers | 1,064 | $8,434.27 |

### Shapiro-Wilk Normality Test

| Group | Statistic | p-value |
|---|---:|---:|
| Smokers | 0.9396 | 3.625 × 10⁻⁹ |
| Non-smokers | 0.8729 | 1.446 × 10⁻²⁸ |

Both p-values are less than 0.05. Therefore, the assumption of normality is rejected for both groups.

### Levene's Test for Equality of Variances

| Statistic | p-value |
|---:|---:|
| 332.6135 | 1.559 × 10⁻⁶⁶ |

The p-value is less than 0.05, indicating that the equal-variance assumption is rejected.

Since the data were non-normal and the variances were unequal, the **Mann-Whitney U test** was used.

### Mann-Whitney U Test

| Statistic | Value |
|---|---:|
| U statistic | 284,133.0 |
| p-value | 5.270 × 10⁻¹³⁰ |

### Conclusion

The p-value is far below 0.05, so the null hypothesis is rejected.

There is a statistically significant difference in medical charges between smokers and non-smokers. Smokers have substantially higher average medical charges ($32,050.23) than non-smokers ($8,434.27).

---

# 🧪 Hypothesis Test 2 — One-Way ANOVA

A One-Way ANOVA was conducted to determine whether mean medical charges differ across the four residential regions.

### Regional Summary

| Region | Count | Mean Charges | Median | Std Dev |
|---|---:|---:|---:|---:|
| Northeast | 324 | $13,406.38 | $10,057.65 | $11,255.80 |
| Northwest | 325 | $12,417.58 | $8,965.80 | $11,072.28 |
| Southeast | 364 | $14,735.41 | $9,294.13 | $13,971.10 |
| Southwest | 325 | $12,346.94 | $8,798.59 | $11,557.18 |

### ANOVA Result

| Statistic | Value |
|---|---:|
| F-statistic | 2.9696 |
| p-value | 0.030893 |
| Significance level (α) | 0.05 |

### Conclusion

Since the p-value (0.030893) is less than 0.05, the null hypothesis is rejected.

There is statistically significant evidence that mean medical charges are not equal across all four regions.

The Southeast had the highest observed mean charges ($14,735.41), while the Southwest had the lowest ($12,346.94).

**Note:** One-Way ANOVA establishes that at least one group mean differs, but it does not identify which specific pairs of regions differ without a post-hoc test.

---

# 📉 Multiple Linear Regression

A multiple linear regression model was developed using:

- `age`
- `bmi`
- `children`
- `sex`
- `smoker`
- `region`

The categorical variables were converted into dummy variables.

### Reference Categories

The reference categories used in the regression model were:

- **Sex:** Female
- **Smoking status:** Non-smoker
- **Region:** Northeast

The regression model was fitted using **Ordinary Least Squares (OLS)**.

## OLS Model Summary

| Statistic | Value |
|---|---:|
| Observations | 1,338 |
| R-squared | 0.751 |
| Adjusted R-squared | 0.749 |
| F-statistic | 500.7 |
| Model p-value | < 0.001 |
| Durbin-Watson | 2.089 |
| Covariance Type | Non-robust |

### OLS Coefficient Results

| Predictor | Coefficient | Std Error | p-value | 95% CI Lower | 95% CI Upper |
|---|---:|---:|---:|---:|---:|
| Intercept | -11,741.43 | 975.29 | <0.001 | -13,654.70* | -9,828.16 |
| Age | 256.98 | 11.90 | <0.001 | 233.64 | 280.32 |
| BMI | 337.99 | 28.52 | <0.001 | 282.05 | 393.93 |
| Children | 478.23 | 137.81 | 0.001 | 207.88 | 748.57 |
| Male | -132.45 | 332.97 | 0.691 | -785.66 | 520.75 |
| Smoker | 23,859.79 | 413.19 | <0.001 | ~23,049* | ~24,671* |
| Northwest | -346.43 | 476.30 | 0.467 | -1,280.82 | 587.96 |
| Southeast | -1,042.41 | 478.90 | 0.030 | -1,981.88 | -102.94 |
| Southwest | -969.56 | 478.03 | 0.043 | -1,907.34 | -31.78 |

*The original statsmodels output displayed some confidence intervals in scientific notation; values shown above are rounded representations where applicable.*

### OLS Interpretation

The regression model explains approximately **75.1% of the variation in medical charges**.

Holding the other variables constant:

- Each additional year of age is associated with approximately **$256.98 higher charges**.
- Each one-unit increase in BMI is associated with approximately **$337.99 higher charges**.
- Each additional child is associated with approximately **$478.23 higher charges**.
- Being a smoker is associated with approximately **$23,859.79 higher charges** compared with being a non-smoker.
- Sex does not have a statistically significant effect in this model.
- The Northwest region is not statistically significant compared with the Northeast.
- Southeast and Southwest have statistically significant negative coefficients relative to the Northeast.

---

# 🔍 Regression Diagnostics

Several diagnostic procedures were used to assess the assumptions and reliability of the regression model.

## Residual Analysis

A **Residual vs Fitted** plot was used to examine whether residuals were randomly distributed around zero and whether the variance remained approximately constant.

A **Q-Q plot** was also used to visually assess the normality of the regression residuals.

## Omnibus / Normality Diagnostic

The OLS model produced an Omnibus statistic of:

- **Omnibus = 299.849**

The diagnostic plots and normality statistics were examined as part of model validation.

---

# 📐 Variance Inflation Factor (VIF)

VIF was calculated to assess multicollinearity among the explanatory variables.

| Variable | VIF |
|---|---:|
| Age | 1.0166 |
| BMI | 1.1079 |
| Children | 1.0040 |
| Sex (Male) | 1.0089 |
| Smoker (Yes) | 1.0121 |
| Region (Northwest) | 1.5188 |
| Region (Southeast) | 1.6534 |
| Region (Southwest) | 1.5299 |

### VIF Interpretation

All VIF values are well below 5, indicating **no serious multicollinearity** among the predictors.

---

# 📊 Breusch-Pagan Test

The Breusch-Pagan test was performed to check for heteroscedasticity.

| Statistic | Value |
|---|---:|
| LM Statistic | 121.9648 |
| LM p-value | 1.302 × 10⁻²² |
| F Statistic | 16.6619 |
| F p-value | 1.021 × 10⁻²³ |

### Conclusion

The p-value is far below 0.05, so the null hypothesis of constant variance is rejected.

Therefore, the regression model shows evidence of **heteroscedasticity**.

To account for this issue, **HC3 heteroscedasticity-robust standard errors** were used for final regression inference.

---

# 🛡️ HC3 Robust Regression

The regression coefficients remain the same, but HC3 provides robust standard errors, p-values, and confidence intervals that are more appropriate when heteroscedasticity is present.

### HC3 Model Summary

| Statistic | Value |
|---|---:|
| Observations | 1,338 |
| R-squared | 0.751 |
| Adjusted R-squared | 0.749 |
| Robust F-statistic | 298.4 |
| Model p-value | 2.47 × 10⁻²⁹⁰ |
| Covariance Type | HC3 |

### HC3 Robust Results

| Predictor | Coefficient | HC3 Std Error | p-value | 95% CI Lower | 95% CI Upper |
|---|---:|---:|---:|---:|---:|
| Intercept | -11,741.43 | 1,035.32 | 1.605 × 10⁻²⁸ | -13,772.47 | -9,710.39 |
| Age | 256.98 | 11.97 | 4.849 × 10⁻⁸⁸ | 233.50 | 280.45 |
| BMI | 337.99 | 31.82 | 2.376 × 10⁻²⁵ | 275.56 | 400.41 |
| Children | 478.23 | 131.01 | 2.721 × 10⁻⁴ | 221.22 | 735.24 |
| Male | -132.45 | 334.98 | 0.693 | -789.60 | 524.69 |
| Smoker | 23,859.79 | 578.17 | 2.736 × 10⁻²⁴⁰ | 22,725.56 | 24,994.02 |
| Northwest | -346.43 | 486.85 | 0.477 | -1,301.52 | 608.65 |
| Southeast | -1,042.41 | 503.35 | 0.039 | -2,029.86 | -54.95 |
| Southwest | -969.56 | 463.34 | 0.037 | -1,878.51 | -60.60 |

### Final Regression Findings

Using HC3 robust standard errors:

- **Age, BMI, children, and smoking status** remain statistically significant predictors.
- **Smoking status has the strongest effect**, with smokers associated with approximately $23,860 higher charges than non-smokers.
- **Sex is not statistically significant.**
- The **Northwest region is not statistically significant** relative to the Northeast.
- The **Southeast and Southwest regions remain statistically significant** relative to the Northeast.
- The model explains approximately **75.1% of the variation** in medical charges.

---

# 🖥️ Streamlit Application

The application is divided into three main tabs.

## 📊 Tab 1 — Data Exploration

Provides:

- Dataset preview
- Number of rows and columns
- Missing-value information
- Interactive age filtering
- Descriptive statistics
- Histograms and KDE plots
- Scatter plots
- Correlation matrix

The plots and statistics react to the selected inputs.

---

## 🧪 Tab 2 — Hypothesis Testing Lab

Provides interactive statistical testing, including:

### Hypothesis Test 1

Comparison of medical charges between:

- Smokers
- Non-smokers

The application performs the appropriate statistical tests and displays the test statistic, p-value, and conclusion.

### Hypothesis Test 2

One-Way ANOVA with:

- **Categorical factor:** Region
- **Numerical variable:** User-selectable numerical variable

The application calculates the ANOVA F-statistic and p-value and provides an automatic conclusion.

---

## 📈 Tab 3 — Live Prediction & Diagnostics

The application provides interactive medical charge prediction.

Users can enter relevant patient characteristics, including:

- Age
- Sex
- BMI-related information
- Number of children
- Smoking status
- Region

### BMI Calculation

BMI can be calculated using height and weight:

```text
BMI = Weight (kg) / Height (m)²