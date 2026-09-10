import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    shapiro,
    levene,
    mannwhitneyu,
    f_oneway
)

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical Insurance Cost Analytics",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       ACTIVE TAB - BLUE
       ======================================================== */

    button[role="tab"][aria-selected="true"] {
        color: #2196F3 !important;
    }

    button[role="tab"][aria-selected="true"] p {
        color: #2196F3 !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #2196F3 !important;
    }

    button[role="tab"]:hover {
        color: #2196F3 !important;
    }

    button[role="tab"]:hover p {
        color: #2196F3 !important;
    }


    /* ========================================================
       SIDEBAR HEADINGS
       ======================================================== */

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #2196F3 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "D:\\DS602_202618011\\insurance.csv"
)


# ============================================================
# OLS REGRESSION MODEL
# ============================================================

X = pd.get_dummies(
    df[
        [
            "age",
            "bmi",
            "children",
            "sex",
            "smoker",
            "region"
        ]
    ],
    drop_first=True
)

X = X.astype(int)

y = df["charges"]

# Add intercept
X = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(
    y,
    X
).fit()


# ============================================================
# SIDEBAR — CONTROL PANEL
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Control Panel"
    )

    st.caption(
        "M.Sc. Data Science — Sem 1"
    )
    st.caption(
            "202618011 - Diya Tilwani"
        )

    st.divider()


    # ========================================================
    # AGE FILTER
    # ========================================================

    st.subheader(
        "🎯 Filter Age Range"
    )

    age_range = st.slider(
        "Select age range",
        min_value=int(df["age"].min()),
        max_value=int(df["age"].max()),
        value=(
            int(df["age"].min()),
            int(df["age"].max())
        )
    )


    st.divider()


    # ========================================================
    # MODEL METRICS
    # ========================================================

    st.subheader(
        "📈 Model Metrics"
    )

    st.metric(
        "Model R²",
        f"{model.rsquared:.3f}"
    )

    st.metric(
        "Adjusted R²",
        f"{model.rsquared_adj:.3f}"
    )

    st.metric(
        "F-Statistic",
        f"{model.fvalue:.2f}"
    )

# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    """
    <style>
    div[data-testid="stHeading"] h1 {
        text-align: center;
    }

    div[data-testid="stCaptionContainer"] {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏥 Medical Insurance Cost Analytics")

st.caption(
    "Statistical Modeling, Hypothesis Testing & Interactive Prediction"
)

st.markdown(
    """
    <div style="text-align: center;">
        Interactive dashboard for exploring medical insurance data,
        performing hypothesis tests, and building statistical models.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Data Exploration",
        "🧪 Hypothesis Testing Lab",
        "📈 Live Prediction & Diagnostics"
    ]
)


# ============================================================
# TAB 1 — DATA EXPLORATION
# ============================================================

with tab1:

    st.header(
        "📊 Data Exploration"
    )


    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(),
        use_container_width=True
    )


    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    st.subheader(
        "Dataset Summary"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )


    # ========================================================
    # DESCRIPTIVE STATISTICS
    # ========================================================

    st.subheader(
        "Descriptive Statistics"
    )

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    descriptive_stats = pd.DataFrame({

        "Mean":
            df[numerical_columns].mean(),

        "Median":
            df[numerical_columns].median(),

        "Standard Deviation":
            df[numerical_columns].std(),

        "IQR":
            (
                df[numerical_columns].quantile(0.75)
                -
                df[numerical_columns].quantile(0.25)
            ),

        "Skewness":
            df[numerical_columns].skew(),

        "Kurtosis":
            df[numerical_columns].kurtosis()

    })

    st.dataframe(
        descriptive_stats.round(3),
        use_container_width=True
    )


    # ========================================================
    # HISTOGRAM / KDE
    # ========================================================

    st.subheader(
        "Distribution of Numerical Variables"
    )

    selected_variable = st.selectbox(
        "Select a variable:",
        numerical_columns,
        key="histogram_variable"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.histplot(
        data=df,
        x=selected_variable,
        kde=True,
        ax=ax
    )

    ax.set_xlabel(
        selected_variable.replace(
            "_",
            " "
        ).title(),
        fontsize=11
    )

    ax.set_ylabel(
        "Frequency",
        fontsize=11
    )

    ax.set_title(
        f"Distribution of "
        f"{selected_variable.replace('_', ' ').title()}",
        fontsize=15,
        pad=15
    )

    ax.tick_params(
        axis="both",
        labelsize=10
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # SCATTER PLOT
    # ========================================================

    st.subheader(
        "Bivariate Scatter Plot"
    )


    # --------------------------------------------------------
    # Apply sidebar AGE filter only
    # --------------------------------------------------------

    filtered_df = df[
        (df["age"] >= age_range[0]) &
        (df["age"] <= age_range[1])
    ]


    # --------------------------------------------------------
    # X and Y variables
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        x_variable = st.selectbox(
            "Select X-axis variable:",
            numerical_columns,
            key="scatter_x"
        )

    with col2:

        y_variable = st.selectbox(
            "Select Y-axis variable:",
            numerical_columns,
            index=3,
            key="scatter_y"
        )


    st.write(
        f"Showing ages from "
        f"**{age_range[0]} to {age_range[1]}**."
    )


    # --------------------------------------------------------
    # Scatter plot
    # --------------------------------------------------------

    if len(filtered_df) > 0:

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        sns.scatterplot(
            data=filtered_df,
            x=x_variable,
            y=y_variable,
            hue="smoker",
            alpha=0.65,
            s=55,
            ax=ax
        )

        ax.set_xlabel(
            x_variable.replace(
                "_",
                " "
            ).title(),
            fontsize=11
        )

        ax.set_ylabel(
            y_variable.replace(
                "_",
                " "
            ).title(),
            fontsize=11
        )

        ax.set_title(
            f"{x_variable.replace('_', ' ').title()} vs "
            f"{y_variable.replace('_', ' ').title()}",
            fontsize=15,
            pad=15
        )

        ax.legend(
            title="Smoker",
            loc="best"
        )

        ax.tick_params(
            axis="both",
            labelsize=10
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.warning(
            "No records match the selected age range."
        )


    # ========================================================
    # CORRELATION MATRIX
    # ========================================================

    st.subheader(
        "Correlation Matrix"
    )

    correlation_matrix = df[
        numerical_columns
    ].corr()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        ax=ax
    )

    ax.set_title(
        "Correlation Between Numerical Variables",
        fontsize=15,
        pad=15
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# TAB 2 — HYPOTHESIS TESTING LAB
# ============================================================

with tab2:

    st.header(
        "🧪 Hypothesis Testing Lab"
    )

    st.write(
        "Test whether medical charges differ "
        "between different groups in the dataset."
    )


    # ========================================================
    # HYPOTHESIS TEST 1
    # ========================================================

    st.subheader(
        "Hypothesis Test 1: Smokers vs Non-Smokers"
    )

    st.write(
        "**Question:** Do smokers and non-smokers have "
        "significantly different medical charges?"
    )

    st.write(
        "**H₀:** There is no significant difference "
        "in medical charges between smokers and non-smokers."
    )

    st.write(
        "**H₁:** There is a significant difference "
        "in medical charges between smokers and non-smokers."
    )


    # ========================================================
    # GROUP STATISTICS
    # ========================================================

    smoker_charges = df[
        df["smoker"] == "yes"
    ]["charges"]

    non_smoker_charges = df[
        df["smoker"] == "no"
    ]["charges"]


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Average Charges - Smokers",
            f"${smoker_charges.mean():,.2f}"
        )

        st.write(
            f"Number of smokers: "
            f"{len(smoker_charges)}"
        )

    with col2:

        st.metric(
            "Average Charges - Non-Smokers",
            f"${non_smoker_charges.mean():,.2f}"
        )

        st.write(
            f"Number of non-smokers: "
            f"{len(non_smoker_charges)}"
        )


    # ========================================================
    # SHAPIRO-WILK TEST
    # ========================================================

    st.subheader(
        "Shapiro-Wilk Normality Test"
    )

    smoker_stat, smoker_p = shapiro(
        smoker_charges
    )

    non_smoker_stat, non_smoker_p = shapiro(
        non_smoker_charges
    )

    normality_results = pd.DataFrame({

        "Group": [
            "Smokers",
            "Non-Smokers"
        ],

        "Test Statistic": [
            smoker_stat,
            non_smoker_stat
        ],

        "P-value": [
            smoker_p,
            non_smoker_p
        ]

    })

    st.dataframe(
        normality_results,
        use_container_width=True
    )

    st.write(
        "If p-value < 0.05, reject H₀ and conclude "
        "that the data is not normally distributed."
    )


    # ========================================================
    # LEVENE'S TEST
    # ========================================================

    st.subheader(
        "Levene's Test for Equal Variances"
    )

    levene_stat, levene_p = levene(
        smoker_charges,
        non_smoker_charges
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Levene Test Statistic",
            f"{levene_stat:.4f}"
        )

    with col2:

        st.metric(
            "P-value",
            f"{levene_p:.4e}"
        )


    if levene_p < 0.05:

        st.error(
            "Reject H₀: The variances of the two groups "
            "are significantly different."
        )

    else:

        st.success(
            "Fail to reject H₀: There is no significant "
            "difference in the variances."
        )


    # ========================================================
    # MANN-WHITNEY U TEST
    # ========================================================

    st.subheader(
        "Mann-Whitney U Test"
    )

    st.write(
        "Since the data is not normally distributed and "
        "the group variances are significantly different, "
        "the Mann-Whitney U test is used."
    )

    u_stat, mann_p = mannwhitneyu(
        smoker_charges,
        non_smoker_charges,
        alternative="two-sided"
    )


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Mann-Whitney U Statistic",
            f"{u_stat:.2f}"
        )

    with col2:

        st.metric(
            "P-value",
            f"{mann_p:.4e}"
        )


    # ========================================================
    # FINAL CONCLUSION
    # ========================================================

    if mann_p < 0.05:

        st.error(
            "Reject H₀: There is a statistically significant "
            "difference in medical charges between smokers "
            "and non-smokers."
        )

    else:

        st.success(
            "Fail to reject H₀: There is no statistically "
            "significant difference in medical charges between "
            "smokers and non-smokers."
        )


    # ========================================================
    # SMOKER BOX PLOT
    # ========================================================

    st.subheader(
        "Medical Charges by Smoking Status"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.boxplot(
        data=df,
        x="smoker",
        y="charges",
        ax=ax
    )

    ax.set_title(
        "Medical Charges: Smokers vs Non-Smokers",
        fontsize=15,
        pad=15
    )

    ax.set_xlabel(
        "Smoking Status",
        fontsize=11
    )

    ax.set_ylabel(
        "Medical Charges ($)",
        fontsize=11
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "Non-Smoker",
            "Smoker"
        ]
    )

    ax.tick_params(
        axis="both",
        labelsize=10
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # HYPOTHESIS TEST 2 — ONE-WAY ANOVA
    # ========================================================

    st.subheader(
        "Hypothesis Test 2: Charges Across Regions"
    )

    st.write(
        "**Question:** Do medical charges differ significantly "
        "across the four regions?"
    )

    st.write(
        "**H₀:** The mean medical charges are equal across "
        "all four regions."
    )

    st.write(
        "**H₁:** At least one region has a different mean "
        "medical charge."
    )


    # ========================================================
    # REGION GROUPS
    # ========================================================

    northeast = df[
        df["region"] == "northeast"
    ]["charges"]

    northwest = df[
        df["region"] == "northwest"
    ]["charges"]

    southeast = df[
        df["region"] == "southeast"
    ]["charges"]

    southwest = df[
        df["region"] == "southwest"
    ]["charges"]


    # ========================================================
    # ONE-WAY ANOVA
    # ========================================================

    f_stat, anova_p = f_oneway(
        northeast,
        northwest,
        southeast,
        southwest
    )


    # ========================================================
    # DISPLAY ANOVA RESULTS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "F-Statistic",
            f"{f_stat:.4f}"
        )

    with col2:

        st.metric(
            "P-value",
            f"{anova_p:.4e}"
        )


    # ========================================================
    # ANOVA CONCLUSION
    # ========================================================

    if anova_p < 0.05:

        st.error(
            "Reject H₀: There is a statistically significant "
            "difference in medical charges across the regions."
        )

    else:

        st.success(
            "Fail to reject H₀: There is no statistically "
            "significant difference in medical charges across "
            "the regions."
        )


    # ========================================================
    # REGION BOX PLOT
    # ========================================================

    st.subheader(
        "Medical Charges by Region"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.boxplot(
        data=df,
        x="region",
        y="charges",
        ax=ax
    )

    ax.set_title(
        "Distribution of Medical Charges by Region",
        fontsize=15,
        pad=15
    )

    ax.set_xlabel(
        "Region",
        fontsize=11
    )

    ax.set_ylabel(
        "Medical Charges ($)",
        fontsize=11
    )

    ax.tick_params(
        axis="both",
        labelsize=10
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# ============================================================

with tab3:

    st.header(
        "🔮 Live Prediction & Diagnostics"
    )

    st.write(
        "Enter the patient's details below to estimate "
        "medical insurance charges."
    )


    # ========================================================
    # LIVE PREDICTION
    # ========================================================

    st.subheader(
        "Predict Medical Charges"
    )

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=60.0
        )

        height = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=250.0,
            value=165.0
        )


        # ----------------------------------------------------
        # Calculate BMI automatically
        # ----------------------------------------------------

        height_m = height / 100

        bmi = weight / (
            height_m ** 2
        )

        st.info(
            f"Calculated BMI: **{bmi:.2f} kg/m²**"
        )


        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0
        )


    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        sex = st.selectbox(
            "Sex",
            [
                "female",
                "male"
            ]
        )

        smoker = st.selectbox(
            "Smoker",
            [
                "no",
                "yes"
            ]
        )

        region = st.selectbox(
            "Region",
            [
                "northeast",
                "northwest",
                "southeast",
                "southwest"
            ]
        )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "⚡ Predict Charges",
        type="primary"
    ):


        # ----------------------------------------------------
        # Create input dataframe
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "age": [age],

            "bmi": [bmi],

            "children": [children],

            "sex": [sex],

            "smoker": [smoker],

            "region": [region]

        })


        # ----------------------------------------------------
        # Convert categorical variables into dummy variables
        # ----------------------------------------------------

        input_data = pd.get_dummies(
            input_data,
            drop_first=True
        )


        # ----------------------------------------------------
        # Match training columns
        # ----------------------------------------------------

        feature_columns = X.columns.drop(
            "const"
        )

        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # ----------------------------------------------------
        # Add intercept
        # ----------------------------------------------------

        input_data = sm.add_constant(
            input_data,
            has_constant="add"
        )


        # ----------------------------------------------------
        # Ensure exact same column order
        # ----------------------------------------------------

        input_data = input_data[
            X.columns
        ]


        # ====================================================
        # PREDICTION + CONFIDENCE INTERVALS
        # ====================================================

        prediction_result = model.get_prediction(
            input_data
        )

        prediction_summary = prediction_result.summary_frame(
            alpha=0.05
        )


        # ----------------------------------------------------
        # Extract values
        # ----------------------------------------------------

        predicted_charge = prediction_summary[
            "mean"
        ].iloc[0]

        confidence_lower = prediction_summary[
            "mean_ci_lower"
        ].iloc[0]

        confidence_upper = prediction_summary[
            "mean_ci_upper"
        ].iloc[0]

        prediction_lower = prediction_summary[
            "obs_ci_lower"
        ].iloc[0]

        prediction_upper = prediction_summary[
            "obs_ci_upper"
        ].iloc[0]


        # ====================================================
        # DISPLAY PREDICTION
        # ====================================================

        st.success(
            f"### Estimated Medical Charges: "
            f"${predicted_charge:,.2f}"
        )


        # ====================================================
        # CONFIDENCE INTERVAL
        # ====================================================

        st.subheader(
            "95% Confidence Interval"
        )

        st.write(
            "This interval estimates the range in which the "
            "average medical charge for patients with these "
            "characteristics is expected to lie."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Lower Bound",
                f"${confidence_lower:,.2f}"
            )

        with col2:

            st.metric(
                "Upper Bound",
                f"${confidence_upper:,.2f}"
            )


        # ====================================================
        # PREDICTION INTERVAL
        # ====================================================

        st.subheader(
            "95% Prediction Interval"
        )

        st.write(
            "This interval gives a wider range for the medical "
            "charge of an individual patient with these "
            "characteristics."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Lower Bound",
                f"${prediction_lower:,.2f}"
            )

        with col2:

            st.metric(
                "Upper Bound",
                f"${prediction_upper:,.2f}"
            )


    # ========================================================
    # MODEL DIAGNOSTICS
    # ========================================================

    st.divider()

    st.header(
        "📊 Model Diagnostics"
    )

    st.write(
        "Diagnostic plots and statistical tests used to "
        "evaluate the assumptions of the OLS regression model."
    )


    # ========================================================
    # RESIDUALS
    # ========================================================

    residuals = model.resid

    fitted_values = model.fittedvalues


    # ========================================================
    # RESIDUAL VS FITTED PLOT
    # ========================================================

    st.subheader(
        "Residuals vs Fitted Values"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.scatterplot(
        x=fitted_values,
        y=residuals,
        alpha=0.6,
        s=45,
        ax=ax
    )

    ax.axhline(
        0,
        linestyle="--"
    )

    ax.set_xlabel(
        "Fitted Values",
        fontsize=11
    )

    ax.set_ylabel(
        "Residuals",
        fontsize=11
    )

    ax.set_title(
        "Residuals vs Fitted Values",
        fontsize=15,
        pad=15
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    st.write(
        "A good residual plot should show points randomly "
        "scattered around zero without a clear pattern."
    )


    # ========================================================
    # Q-Q PLOT
    # ========================================================

    st.subheader(
        "Q-Q Plot of Residuals"
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    sm.qqplot(
        residuals,
        line="45",
        ax=ax
    )

    ax.set_title(
        "Normal Q-Q Plot",
        fontsize=15,
        pad=15
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    st.write(
        "If the points approximately follow the diagonal "
        "line, the residuals are closer to normally distributed."
    )


    # ========================================================
    # JARQUE-BERA TEST
    # ========================================================

    st.subheader(
        "Jarque-Bera Normality Test"
    )

    jb_stat, jb_pvalue, skewness, kurtosis = (
        sm.stats.jarque_bera(
            residuals
        )
    )


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Jarque-Bera Statistic",
            f"{jb_stat:.4f}"
        )

    with col2:

        st.metric(
            "P-value",
            f"{jb_pvalue:.4e}"
        )


    if jb_pvalue < 0.05:

        st.warning(
            "Reject H₀: The residuals are not normally distributed."
        )

    else:

        st.success(
            "Fail to reject H₀: There is no significant evidence "
            "against normality of the residuals."
        )


    # ========================================================
    # BREUSCH-PAGAN TEST
    # ========================================================

    st.subheader(
        "Breusch-Pagan Test for Heteroscedasticity"
    )

    bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = (
        het_breuschpagan(
            residuals,
            model.model.exog
        )
    )


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "LM Statistic",
            f"{bp_lm:.4f}"
        )

    with col2:

        st.metric(
            "P-value",
            f"{bp_lm_pvalue:.4e}"
        )


    if bp_lm_pvalue < 0.05:

        st.warning(
            "Reject H₀: Significant heteroscedasticity "
            "is present in the residuals."
        )

    else:

        st.success(
            "Fail to reject H₀: There is no significant "
            "evidence of heteroscedasticity."
        )


    # ========================================================
    # VIF
    # ========================================================

    st.subheader(
        "Variance Inflation Factor (VIF)"
    )

    st.write(
        "VIF is used to detect multicollinearity among "
        "the explanatory variables."
    )


    X_vif = X.drop(
        columns=["const"]
    )


    vif_data = pd.DataFrame({

        "Variable":
            X_vif.columns,

        "VIF":
            [
                variance_inflation_factor(
                    X_vif.values,
                    i
                )
                for i in range(
                    X_vif.shape[1]
                )
            ]

    })


    st.dataframe(
        vif_data.round(3),
        use_container_width=True
    )


    st.info(
        "As a general rule, VIF values below 5 indicate "
        "low to moderate multicollinearity."
    )


    # ========================================================
# OLS MODEL SUMMARY
# ========================================================

st.divider()

st.subheader(
    "📄 OLS Regression Results"
)

st.write(
    "Multiple linear regression model used to estimate "
    "medical insurance charges."
)


# ========================================================
# MODEL PERFORMANCE METRICS
# ========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "R²",
        f"{model.rsquared:.3f}"
    )

with col2:

    st.metric(
        "Adjusted R²",
        f"{model.rsquared_adj:.3f}"
    )

with col3:

    st.metric(
        "F-Statistic",
        f"{model.fvalue:.2f}"
    )

with col4:

    st.metric(
        "Observations",
        f"{int(model.nobs)}"
    )


# ========================================================
# OVERALL MODEL SIGNIFICANCE
# ========================================================

if model.f_pvalue < 0.05:

    st.success(
        "Overall Model: Significant "
        "(p < 0.05)"
    )

else:

    st.info(
        "Overall Model: Not statistically significant "
        "(p ≥ 0.05)"
    )


# ========================================================
# COEFFICIENT TABLE
# ========================================================

st.subheader(
    "Regression Coefficients"
)


# Create coefficient table

coef_table = pd.DataFrame({

    "Coefficient": model.params,

    "Std. Error": model.bse,

    "t-Statistic": model.tvalues,

    "P-value": model.pvalues,

    "95% CI Lower": model.conf_int()[0],

    "95% CI Upper": model.conf_int()[1]

})


# Round values for clean display

coef_table = coef_table.round(3)


st.dataframe(
    coef_table,
    use_container_width=True
)


# ========================================================
# SIGNIFICANCE INTERPRETATION
# ========================================================

st.subheader(
    "📌 Key Findings"
)


# Find significant variables

significant_variables = model.pvalues[
    model.pvalues < 0.05
].index.tolist()


if len(significant_variables) > 0:

    st.write(
        "The following variables are statistically "
        "significant at α = 0.05:"
    )

    for variable in significant_variables:

        if variable == "const":
            continue

        coefficient = model.params[variable]

        if coefficient > 0:

            st.write(
                f"• **{variable}** has a positive association "
                f"with medical charges "
                f"(coefficient = {coefficient:,.2f})."
            )

        else:

            st.write(
                f"• **{variable}** has a negative association "
                f"with medical charges "
                f"(coefficient = {coefficient:,.2f})."
            )


# ========================================================
# MODEL EQUATION
# ========================================================

with st.expander(
    "📐 View Regression Model Details" 
):

    st.write(
        "**Dependent variable:** `charges`"
    )

    st.write(
        "**Reference categories:** "
        "Female, Non-smoker, Northeast"
    )

    st.write(
        "Categorical variables were converted into dummy "
        "variables using `drop_first=True`."
    )

    st.write(
        "The model estimates medical charges using age, "
        "BMI, number of children, sex, smoking status, "
        "and region."
    )