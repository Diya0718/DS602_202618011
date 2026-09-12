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
    "D:\\Diya\\insurance.csv"
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

    st.header("🧪 Hypothesis Testing Lab")
    st.write(
        "Select variables below to perform statistical hypothesis tests "
        "interactively at α = 0.05."
    )

    alpha = 0.05

    # --------------------------------------------------------
    # HYPOTHESIS TEST 1 — TWO GROUP COMPARISON
    # --------------------------------------------------------

    st.subheader("🔬 Hypothesis Test 1 — Compare Two Groups")

    st.write(
        "**Purpose:** Determine whether the selected numerical variable "
        "differs significantly between two groups."
    )

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        ht1_category = st.selectbox(
            "Select categorical factor",
            categorical_columns,
            key="ht1_category"
        )

    with col2:
        ht1_numeric = st.selectbox(
            "Select numerical metric",
            numerical_columns,
            key="ht1_numeric"
        )

    # Find groups automatically
    ht1_groups = df[ht1_category].dropna().unique().tolist()

    if len(ht1_groups) >= 2:

        group1 = ht1_groups[0]
        group2 = ht1_groups[1]

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**Group 1:** {group1}")

        with col2:
            st.info(f"**Group 2:** {group2}")

        data1 = df[df[ht1_category] == group1][ht1_numeric].dropna()
        data2 = df[df[ht1_category] == group2][ht1_numeric].dropna()

        # Hypotheses
        st.markdown("### Hypotheses")

        st.write(
            f"**H₀:** There is no significant difference in "
            f"**{ht1_numeric}** between the two groups."
        )

        st.write(
            f"**H₁:** There is a significant difference in "
            f"**{ht1_numeric}** between the two groups."
        )

        # ----------------------------------------------------
        # Shapiro-Wilk Normality Test
        # ----------------------------------------------------

        st.markdown("### 1️⃣ Shapiro-Wilk Normality Test")

        shapiro1_stat, shapiro1_p = shapiro(data1)
        shapiro2_stat, shapiro2_p = shapiro(data2)

        normality_table = pd.DataFrame({
            "Group": [str(group1), str(group2)],
            "Statistic": [shapiro1_stat, shapiro2_stat],
            "P-value": [shapiro1_p, shapiro2_p]
        })

        st.dataframe(
            normality_table.style.format({
                "Statistic": "{:.4f}",
                "P-value": "{:.6g}"
            }),
            use_container_width=True
        )

        normal1 = shapiro1_p > alpha
        normal2 = shapiro2_p > alpha

        if normal1 and normal2:
            st.success(
                "Both groups are approximately normally distributed "
                "(p > 0.05)."
            )
        else:
            st.warning(
                "At least one group is not normally distributed "
                "(p ≤ 0.05)."
            )

        # ----------------------------------------------------
        # Levene's Test
        # ----------------------------------------------------

        st.markdown("### 2️⃣ Levene's Test for Equal Variance")

        levene_stat, levene_p = levene(data1, data2)

        st.metric(
            "Levene p-value",
            f"{levene_p:.6g}"
        )

        equal_variance = levene_p > alpha

        if equal_variance:
            st.success(
                "The variances can be considered equal (p > 0.05)."
            )
        else:
            st.warning(
                "The variances are significantly different (p ≤ 0.05)."
            )

        # ----------------------------------------------------
        # Automatically Select Test
        # ----------------------------------------------------

        st.markdown("### 3️⃣ Final Statistical Test")

        if normal1 and normal2:

            # Both groups are normal → t-test
            test_stat, test_p = ttest_ind(
                data1,
                data2,
                equal_var=equal_variance
            )

            test_name = "Independent Two-Sample t-test"

            st.info(
                "Both groups are approximately normal, so an "
                "**Independent Two-Sample t-test** is used."
            )

        else:

            # At least one group is non-normal → Mann-Whitney
            test_stat, test_p = mannwhitneyu(
                data1,
                data2,
                alternative="two-sided"
            )

            test_name = "Mann-Whitney U test"

            st.info(
                "At least one group is non-normal, so the "
                "**Mann-Whitney U test** is used."
            )

        # ----------------------------------------------------
        # Test Results
        # ----------------------------------------------------

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Test",
                test_name
            )

        with result_col2:
            st.metric(
                "P-value",
                f"{test_p:.6g}"
            )

        st.write(
            f"**Test Statistic:** {test_stat:.6f}"
        )

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        if test_p < alpha:

            st.error(
                "❌ **Decision: Reject H₀**"
            )

            st.write(
                f"At α = {alpha}, there is statistically significant "
                f"evidence that **{ht1_numeric} differs between "
                f"{group1} and {group2}**."
            )

        else:

            st.success(
                "✅ **Decision: Fail to Reject H₀**"
            )

            st.write(
                f"At α = {alpha}, there is not enough evidence to conclude "
                f"that **{ht1_numeric} differs between {group1} and {group2}**."
            )

        # ----------------------------------------------------
        # Group Comparison Plot
        # ----------------------------------------------------

        st.markdown("### 📊 Group Comparison")

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x=ht1_category,
            y=ht1_numeric,
            ax=ax
        )

        ax.set_title(
            f"{ht1_numeric} by {ht1_category}"
        )

        ax.set_xlabel(ht1_category)
        ax.set_ylabel(ht1_numeric)

        plt.xticks(rotation=20)

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.warning(
            "The selected categorical variable must contain at least "
            "two groups for Hypothesis Test 1."
        )

    # ========================================================
    # HYPOTHESIS TEST 2 — ONE-WAY ANOVA
    # ========================================================

    st.divider()

    st.subheader("📈 Hypothesis Test 2 — One-Way ANOVA")

    st.write(
        "**Purpose:** Determine whether the selected numerical variable "
        "differs significantly across three or more groups."
    )

    col1, col2 = st.columns(2)

    with col1:
        ht2_category = st.text_input(
            "Categorical factor",
            value="region",
            disabled=True
        )
        

    with col2:
        ht2_numeric = st.selectbox(
            "Select numerical metric",
            numerical_columns,
            key="ht2_numeric"
        )

    # Get groups
    ht2_groups = df[ht2_category].dropna().unique().tolist()

    if len(ht2_groups) >= 3:

        # Create groups for ANOVA
        anova_groups = []

        valid_group_names = []

        for group in ht2_groups:

            group_data = df[
                df[ht2_category] == group
            ][ht2_numeric].dropna()

            if len(group_data) > 1:

                anova_groups.append(group_data)
                valid_group_names.append(group)

        if len(anova_groups) >= 3:

            # ------------------------------------------------
            # Hypotheses
            # ------------------------------------------------

            st.markdown("### Hypotheses")

            st.write(
                f"**H₀:** The mean **{ht2_numeric}** is equal "
                f"across all groups of **{ht2_category}**."
            )

            st.write(
                f"**H₁:** At least one group has a different mean "
                f"**{ht2_numeric}**."
            )

            # ------------------------------------------------
            # ANOVA
            # ------------------------------------------------

            f_stat, anova_p = f_oneway(*anova_groups)

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "F-statistic",
                    f"{f_stat:.6f}"
                )

            with result_col2:
                st.metric(
                    "P-value",
                    f"{anova_p:.6g}"
                )

            # ------------------------------------------------
            # Decision
            # ------------------------------------------------

            if anova_p < alpha:

                st.error(
                    "❌ **Decision: Reject H₀**"
                )

                st.write(
                    f"At α = {alpha}, there is statistically significant "
                    f"evidence that the mean **{ht2_numeric}** is not the "
                    f"same across all **{ht2_category}** groups."
                )

            else:

                st.success(
                    "✅ **Decision: Fail to Reject H₀**"
                )

                st.write(
                    f"At α = {alpha}, there is not enough evidence to "
                    f"conclude that the mean **{ht2_numeric}** differs "
                    f"across the **{ht2_category}** groups."
                )

            # ------------------------------------------------
            # Group Means
            # ------------------------------------------------

            st.markdown("### 📊 Group Means")

            group_means = (
                df.groupby(ht2_category)[ht2_numeric]
                .agg(["mean", "count"])
                .reset_index()
            )

            group_means.columns = [
                ht2_category,
                "Mean",
                "Count"
            ]

            st.dataframe(
                group_means.style.format({
                    "Mean": "{:.2f}",
                    "Count": "{:.0f}"
                }),
                use_container_width=True
            )

            # ------------------------------------------------
            # ANOVA Boxplot
            # ------------------------------------------------

            st.markdown("### 📦 Distribution Across Groups")

            fig, ax = plt.subplots(figsize=(9, 5))

            sns.boxplot(
                data=df,
                x=ht2_category,
                y=ht2_numeric,
                ax=ax
            )

            ax.set_title(
                f"{ht2_numeric} by {ht2_category}"
            )

            ax.set_xlabel(ht2_category)
            ax.set_ylabel(ht2_numeric)

            plt.xticks(rotation=20)

            st.pyplot(fig)

            plt.close(fig)

        else:

            st.warning(
                "At least three groups with sufficient observations "
                "are required for ANOVA."
            )

    else:

        st.warning(
            "The selected categorical variable must contain at least "
            "three groups for One-Way ANOVA."
        )


# ============================================================
# TAB 3: LIVE PREDICTION & DIAGNOSTICS
# ============================================================

with tab3:

    st.header("📈 Live Prediction & Diagnostics")
    st.write(
        "Enter patient information below to predict medical insurance "
        "charges and examine the regression model diagnostics."
    )

    # ========================================================
    # SECTION 1: LIVE PREDICTION
    # ========================================================

    st.subheader("🔮 Live Medical Charge Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        pred_age = st.slider(
            "Age",
            min_value=int(df["age"].min()),
            max_value=int(df["age"].max()),
            value=30
        )

        pred_sex = st.selectbox(
            "Sex",
            sorted(df["sex"].unique())
        )

        pred_children = st.slider(
            "Number of children",
            min_value=int(df["children"].min()),
            max_value=int(df["children"].max()),
            value=0
        )

    with col2:
        pred_weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=200.0,
            value=65.0,
            step=1.0
        )

        pred_height = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=220.0,
            value=165.0,
            step=1.0
        )

        # Calculate BMI
        height_m = pred_height / 100
        pred_bmi = pred_weight / (height_m ** 2)

        st.metric(
            "Calculated BMI",
            f"{pred_bmi:.2f}"
        )

    with col3:
        pred_smoker = st.selectbox(
            "Smoker",
            sorted(df["smoker"].unique())
        )

        pred_region = st.selectbox(
            "Region",
            sorted(df["region"].unique())
        )

    # ========================================================
    # CREATE INPUT DATA FOR MODEL
    # ========================================================

    input_data = pd.DataFrame({
        "age": [pred_age],
        "bmi": [pred_bmi],
        "children": [pred_children],
        "sex": [pred_sex],
        "smoker": [pred_smoker],
        "region": [pred_region]
    })

    # Convert categorical variables into dummy variables
    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )

    # Make sure input columns match the training model
    feature_columns = X.columns.drop("const")

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Add constant
    input_data = sm.add_constant(
        input_data,
        has_constant="add"
    )

    # Ensure exact same column order as training data
    input_data = input_data[X.columns]

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction_result = model.get_prediction(input_data)

    prediction_summary = prediction_result.summary_frame(
        alpha=0.05
    )

    predicted_charge = prediction_summary["mean"].iloc[0]

    confidence_lower = prediction_summary["mean_ci_lower"].iloc[0]
    confidence_upper = prediction_summary["mean_ci_upper"].iloc[0]

    prediction_lower = prediction_summary["obs_ci_lower"].iloc[0]
    prediction_upper = prediction_summary["obs_ci_upper"].iloc[0]

    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    st.divider()

    st.subheader("💰 Predicted Insurance Charge")

    st.metric(
        "Predicted Medical Charge",
        f"${predicted_charge:,.2f}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"""
            **95% Confidence Interval**

            ${confidence_lower:,.2f} to ${confidence_upper:,.2f}

            This interval estimates the uncertainty around the
            **mean predicted charge** for patients with these characteristics.
            """
        )

    with col2:
        st.warning(
            f"""
            **95% Prediction Interval**

            ${prediction_lower:,.2f} to ${prediction_upper:,.2f}

            This wider interval represents the expected range for an
            **individual patient's actual charge**.
            """
        )

    # ========================================================
    # PATIENT INPUT SUMMARY
    # ========================================================

    with st.expander("📋 View Patient Input"):

        patient_summary = pd.DataFrame({
            "Variable": [
                "Age",
                "Sex",
                "Weight",
                "Height",
                "BMI",
                "Children",
                "Smoker",
                "Region"
            ],
            "Value": [
                pred_age,
                pred_sex,
                f"{pred_weight:.1f} kg",
                f"{pred_height:.1f} cm",
                f"{pred_bmi:.2f}",
                pred_children,
                pred_smoker,
                pred_region
            ]
        })

        st.dataframe(
            patient_summary,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # SECTION 2: RESIDUAL DIAGNOSTICS
    # ========================================================

    st.divider()

    st.subheader("🧪 Regression Diagnostics")

    residuals = model.resid
    fitted_values = model.fittedvalues

    # --------------------------------------------------------
    # Residuals vs Fitted
    # --------------------------------------------------------

    st.markdown("### 1. Residuals vs Fitted Values")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        x=fitted_values,
        y=residuals,
        alpha=0.5,
        ax=ax
    )

    ax.axhline(
        0,
        linestyle="--"
    )

    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted Values")

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        "A random scatter around zero supports the linearity and "
        "constant-variance assumptions. A visible pattern or funnel "
        "shape may indicate model problems."
    )

    # --------------------------------------------------------
    # Q-Q Plot
    # --------------------------------------------------------

    st.markdown("### 2. Q-Q Plot")

    fig, ax = plt.subplots(figsize=(7, 5))

    sm.qqplot(
        residuals,
        line="45",
        ax=ax
    )

    ax.set_title("Normal Q-Q Plot of Residuals")

    st.pyplot(fig)

    plt.close(fig)

    st.write(
        "Points approximately following the diagonal line indicate "
        "that the residuals are reasonably close to normally distributed."
    )

    # ========================================================
    # JARQUE-BERA TEST
    # ========================================================

    st.markdown("### 3. Jarque–Bera Normality Test")

    jb_stat, jb_pvalue, skewness, kurtosis = sm.stats.jarque_bera(
        residuals
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "JB Statistic",
            f"{jb_stat:.4f}"
        )

    with col2:
        st.metric(
            "p-value",
            f"{jb_pvalue:.4e}"
        )

    with col3:
        st.metric(
            "Skewness",
            f"{skewness:.4f}"
        )

    with col4:
        st.metric(
            "Kurtosis",
            f"{kurtosis:.4f}"
        )

    if jb_pvalue < 0.05:
        st.warning(
            "Reject H₀: the residuals are not normally distributed "
            "at the 5% significance level."
        )
    else:
        st.success(
            "Fail to reject H₀: there is insufficient evidence "
            "that the residuals are non-normal."
        )

    # ========================================================
    # BREUSCH-PAGAN TEST
    # ========================================================

    st.markdown("### 4. Breusch–Pagan Test for Heteroscedasticity")

    bp_test = het_breuschpagan(
        residuals,
        model.model.exog
    )

    bp_lm_stat = bp_test[0]
    bp_lm_pvalue = bp_test[1]
    bp_f_stat = bp_test[2]
    bp_f_pvalue = bp_test[3]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "LM Statistic",
            f"{bp_lm_stat:.4f}"
        )

    with col2:
        st.metric(
            "LM p-value",
            f"{bp_lm_pvalue:.4e}"
        )

    with col3:
        st.metric(
            "F Statistic",
            f"{bp_f_stat:.4f}"
        )

    with col4:
        st.metric(
            "F p-value",
            f"{bp_f_pvalue:.4e}"
        )

    if bp_lm_pvalue < 0.05:
        st.warning(
            "Reject H₀: significant heteroscedasticity is detected. "
            "The residual variance is not constant."
        )
    else:
        st.success(
            "Fail to reject H₀: there is insufficient evidence "
            "of heteroscedasticity."
        )

    # ========================================================
    # VIF
    # ========================================================

    st.markdown("### 5. Variance Inflation Factor (VIF)")

    X_vif = X.drop(
        columns=["const"]
    )

    vif_data = pd.DataFrame()

    vif_data["Feature"] = X_vif.columns

    vif_data["VIF"] = [
        variance_inflation_factor(
            X_vif.values,
            i
        )
        for i in range(X_vif.shape[1])
    ]

    st.dataframe(
        vif_data,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "VIF values close to 1 indicate little multicollinearity. "
        "Higher values indicate stronger correlation among predictors."
    )

    # ========================================================
    # SECTION 3: MODEL PERFORMANCE
    # ========================================================

    st.divider()

    st.subheader("📊 Multiple Linear Regression Results")

    # --------------------------------------------------------
    # Model metrics
    # --------------------------------------------------------

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
            "F-statistic",
            f"{model.fvalue:.2f}"
        )

    with col4:
        st.metric(
            "Observations",
            f"{int(model.nobs)}"
        )

    # --------------------------------------------------------
    # Overall model significance
    # --------------------------------------------------------

    if model.f_pvalue < 0.05:
        st.success(
            f"Overall model is statistically significant "
            f"(F-test p-value = {model.f_pvalue:.4e})."
        )
    else:
        st.warning(
            f"Overall model is not statistically significant "
            f"(F-test p-value = {model.f_pvalue:.4e})."
        )

    # ========================================================
    # COEFFICIENT TABLE
    # ========================================================

    st.markdown("### Coefficient Estimates")

    coefficient_table = pd.DataFrame({
        "Coefficient": model.params,
        "Std Error": model.bse,
        "t-statistic": model.tvalues,
        "P-value": model.pvalues,
        "CI Lower": model.conf_int()[0],
        "CI Upper": model.conf_int()[1]
    })

    coefficient_table = coefficient_table.round(4)

    st.dataframe(
        coefficient_table,
        use_container_width=True
    )

    # ========================================================
    # KEY FINDINGS
    # ========================================================

    st.markdown("### 🔍 Key Findings")

    st.write(
        f"""
        - The regression model explains approximately
          **{model.rsquared * 100:.1f}%** of the variation in medical charges.
        - The adjusted R² is **{model.rsquared_adj * 100:.1f}%**.
        - The overall regression model is statistically significant.
        - The coefficient estimates show the expected change in medical
          charges for a one-unit increase in a predictor while holding
          the other predictors constant.
        """
    )

    # ========================================================
# OLS REGRESSION RESULTS
# ========================================================

st.subheader("OLS Regression Results")

# Model summary metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("R²", f"{model.rsquared:.3f}")

with col2:
    st.metric("Adjusted R²", f"{model.rsquared_adj:.3f}")

with col3:
    st.metric("F-statistic", f"{model.fvalue:.2f}")

with col4:
    st.metric("Observations", int(model.nobs))


# Coefficient table
ols_table = pd.DataFrame({
    "Coefficient": model.params,
    "Std Error": model.bse,
    "t-statistic": model.tvalues,
    "P-value": model.pvalues,
    "CI Lower": model.conf_int()[0],
    "CI Upper": model.conf_int()[1]
})

st.markdown("### Coefficient Estimates")

st.dataframe(
    ols_table.style.format({
        "Coefficient": "{:.2f}",
        "Std Error": "{:.2f}",
        "t-statistic": "{:.2f}",
        "P-value": "{:.4e}",
        "CI Lower": "{:.2f}",
        "CI Upper": "{:.2f}"
    }),
    use_container_width=True
)
