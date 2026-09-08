import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, levene, mannwhitneyu, f_oneway

import statsmodels.api as sm


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical Insurance Statistical Analysis",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("D:\\DS602_202618011\\insurance.csv")
# OLS Regression Model

X = pd.get_dummies(
    df[["age", "bmi", "children", "sex", "smoker", "region"]],
    drop_first=True
)

X = X.astype(int)

y = df["charges"]

sm.add_constant(X)

model = sm.OLS(y, X).fit()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div style="text-align: center; padding: 20px 0 25px 0;">
        <h1 style="font-size: 42px; margin-bottom: 8px;">
            🏥 Medical Insurance Cost Analytics
        </h1>
        <p style="font-size: 18px; margin-top: 0;">
            Statistical Modeling, Hypothesis Testing & Interactive Prediction
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Interactive dashboard for exploring medical insurance data, "
    "performing hypothesis tests, and building statistical models."
)


# ============================================================
# CREATE TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Data Exploration",
    "🧪 Hypothesis Testing Lab",
    "📈 Live Prediction & Diagnostics"
])


# ============================================================
# TAB 1 — DATA EXPLORATION
# ============================================================

with tab1:

    st.header("📊 Data Exploration")

    # --------------------------------------------------------
    # Dataset Preview
    # --------------------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )


    # --------------------------------------------------------
    # Dataset Information
    # --------------------------------------------------------

    st.subheader("Dataset Information")

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


    # --------------------------------------------------------
    # Descriptive Statistics
    # --------------------------------------------------------

    st.subheader("Descriptive Statistics")

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


        # --------------------------------------------------------
    # Histogram / KDE
    # --------------------------------------------------------

    st.subheader("Distribution of Numerical Variables")

    selected_variable = st.selectbox(
        "Select a variable:",
        numerical_columns,
        key="histogram_variable"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        data=df,
        x=selected_variable,
        kde=True,
        ax=ax
    )

    ax.set_xlabel(
        selected_variable.replace("_", " ").title(),
        fontsize=11
    )

    ax.set_ylabel(
        "Frequency",
        fontsize=11
    )

    ax.set_title(
        f"Distribution of {selected_variable.replace('_', ' ').title()}",
        fontsize=15,
        pad=15
    )

    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # --------------------------------------------------------
    # Scatter Plot
    # --------------------------------------------------------

    st.subheader("Bivariate Scatter Plot")

    # Age slider
    age_range = st.slider(
        "Select Age Range:",
        min_value=int(df["age"].min()),
        max_value=int(df["age"].max()),
        value=(
            int(df["age"].min()),
            int(df["age"].max())
        )
    )

    # Filter data based on selected age range
    filtered_df = df[
        (df["age"] >= age_range[0]) &
        (df["age"] <= age_range[1])
    ]

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
        f"Showing ages from **{age_range[0]} to {age_range[1]}**"
    )

    fig, ax = plt.subplots(figsize=(9, 5))

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
        x_variable.replace("_", " ").title(),
        fontsize=11
    )

    ax.set_ylabel(
        y_variable.replace("_", " ").title(),
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


    # --------------------------------------------------------
    # Correlation Matrix
    # --------------------------------------------------------

    st.subheader("Correlation Matrix")

    correlation_matrix = df[numerical_columns].corr()

    fig, ax = plt.subplots(figsize=(8, 6))

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
# TAB 2 — HYPOTHESIS TESTING
# ============================================================

with tab2:

    st.header("🧪 Hypothesis Testing Lab")

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


    # --------------------------------------------------------
    # Group Statistics
    # --------------------------------------------------------

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
            f"Number of smokers: {len(smoker_charges)}"
        )

    with col2:

        st.metric(
            "Average Charges - Non-Smokers",
            f"${non_smoker_charges.mean():,.2f}"
        )

        st.write(
            f"Number of non-smokers: {len(non_smoker_charges)}"
        )


    # --------------------------------------------------------
    # Shapiro-Wilk Normality Test
    # --------------------------------------------------------

    st.subheader(
        "Shapiro-Wilk Normality Test"
    )

    smoker_stat, smoker_p = shapiro(
        smoker_charges
    )

    non_smoker_stat, non_smoker_p = shapiro(
        non_smoker_charges
    )

    results = pd.DataFrame({

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
        results,
        use_container_width=True
    )

    st.write(
        "If p-value < 0.05, we reject H₀ and conclude "
        "that the data is not normally distributed."
    )


    # --------------------------------------------------------
    # Levene's Test for Equal Variances
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Mann-Whitney U Test
    # --------------------------------------------------------

    st.subheader(
        "Mann-Whitney U Test"
    )

    st.write(
        "Since the data is not normally distributed and the "
        "group variances are significantly different, "
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


    # --------------------------------------------------------
    # Final Conclusion
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Box Plot: Smokers vs Non-Smokers
    # --------------------------------------------------------

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

    ax.set_xticklabels([
        "Non-Smoker",
        "Smoker"
    ])

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


    # --------------------------------------------------------
    # Separate Charges by Region
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Perform One-Way ANOVA
    # --------------------------------------------------------

    f_stat, anova_p = f_oneway(
        northeast,
        northwest,
        southeast,
        southwest
    )


    # --------------------------------------------------------
    # Display Results
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # ANOVA Conclusion
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Region Box Plot
    # --------------------------------------------------------

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
# TAB 3 — PREDICTION & DIAGNOSTICS
# ============================================================

with tab3:

    st.header("🔮 Live Prediction & Diagnostics")

    st.write(
        "Enter the patient's details below to estimate "
        "medical insurance charges."
    )


    # ========================================================
    # LIVE PREDICTION
    # ========================================================

    st.subheader("Predict Medical Charges")


    # --------------------------------------------------------
    # Patient Inputs
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # Left Column
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

        # Calculate BMI automatically
        height_m = height / 100

        bmi = weight / (height_m ** 2)

        st.info(
            f"Calculated BMI: **{bmi:.2f}**"
        )

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0
        )


    # --------------------------------------------------------
    # Right Column
    # --------------------------------------------------------

    with col2:

        sex = st.selectbox(
            "Sex",
            ["female", "male"]
        )

        smoker = st.selectbox(
            "Smoker",
            ["no", "yes"]
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
        "Predict Charges",
        type="primary"
    ):

        # Create input dataframe
        input_data = pd.DataFrame({
            "age": [age],
            "bmi": [bmi],
            "children": [children],
            "sex": [sex],
            "smoker": [smoker],
            "region": [region]
        })


        # Convert categorical variables into dummy variables
        input_data = pd.get_dummies(
            input_data,
            drop_first=True
        )


        # ----------------------------------------------------
        # Match input columns with training columns
        # ----------------------------------------------------

        feature_columns = X.columns.drop("const")

        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # ----------------------------------------------------
        # Add intercept / constant
        # ----------------------------------------------------

        input_data = sm.add_constant(
            input_data,
            has_constant="add"
        )


        # Make sure column order is exactly the same
        input_data = input_data[X.columns]


        # ----------------------------------------------------
        # Predict medical charges
        # ----------------------------------------------------

        predicted_charge = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.success(
            f"### Estimated Medical Charges: "
            f"${predicted_charge:,.2f}"
        )