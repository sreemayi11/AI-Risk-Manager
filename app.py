import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Risk Manager",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "model/best_fraud_model.joblib"
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_test_data():

    X_test = pd.read_csv(
        "data/X_test.csv"
    )

    y_test = pd.read_csv(
        "data/y_test.csv"
    ).squeeze()

    return X_test, y_test


# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

@st.cache_data
def load_feature_importance():

    return pd.read_csv(
        "model/feature_importance.csv"
    )


# ============================================================
# LOAD FINAL METRICS
# ============================================================

@st.cache_data
def load_final_metrics():

    return pd.read_csv(
        "model/final_test_metrics.csv"
    )


# ============================================================
# INITIALIZE
# ============================================================

model = load_model()

X_test, y_test = load_test_data()

feature_importance = load_feature_importance()

final_metrics = load_final_metrics()


with open(
    "model/best_threshold.txt",
    "r"
) as file:

    threshold = float(
        file.read().strip()
    )


# ============================================================
# TITLE
# ============================================================

st.title("🛡️ AI Risk Manager")

st.subheader(
    "Cost-Aware Payment Fraud Detection"
)

st.write(
    """
    A defense-oriented machine learning system that evaluates
    payment transactions, estimates fraud risk, and recommends
    whether a transaction should be allowed or manually reviewed.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ System Information"
)

st.sidebar.write(
    f"**Model:** {type(model).__name__}"
)

st.sidebar.write(
    f"**Decision Threshold:** {threshold:.2f}"
)

st.sidebar.write(
    "**Actions:** ALLOW / REVIEW"
)

st.sidebar.divider()

st.sidebar.info(
    """
    V1–V28 are anonymized PCA-based features
    from the public fraud detection dataset.

    Their real-world meanings are not assumed.
    """
)


# ============================================================
# SAMPLE TRANSACTION SELECTION
# ============================================================

st.header(
    "💳 Transaction Risk Assessment"
)

st.write(
    "Select a transaction scenario to demonstrate the risk assessment system."
)


transaction_type = st.radio(
    "Transaction Input Mode",
    [
        "🟢 Sample Legitimate Transaction",
        "🔴 Sample Fraud Transaction",
        "✏️ Manual Transaction"
    ],
    horizontal=True
)


# ============================================================
# SELECT SAMPLE TRANSACTION
# ============================================================

if transaction_type == "🟢 Sample Legitimate Transaction":

    legitimate_indices = y_test[
        y_test == 0
    ].index

    selected_index = legitimate_indices[0]

    selected_transaction = X_test.loc[
        selected_index
    ].copy()

    actual_class = 0


elif transaction_type == "🔴 Sample Fraud Transaction":

    fraud_indices = y_test[
        y_test == 1
    ].index

    selected_index = fraud_indices[0]

    selected_transaction = X_test.loc[
        selected_index
    ].copy()

    actual_class = 1


else:

    selected_transaction = None

    actual_class = None


# ============================================================
# MANUAL INPUT
# ============================================================

if transaction_type == "✏️ Manual Transaction":

    st.subheader(
        "Enter Transaction Features"
    )

    col1, col2 = st.columns(2)

    with col1:

        time_value = st.number_input(
            "Time",
            min_value=0.0,
            value=10000.0,
            step=1.0
        )

    with col2:

        amount_value = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

    feature_values = {}

    columns = st.columns(4)

    for i in range(1, 29):

        feature_name = f"V{i}"

        column = columns[
            (i - 1) % 4
        ]

        with column:

            feature_values[feature_name] = st.number_input(
                feature_name,
                value=0.0,
                format="%.6f",
                key=feature_name
            )

    input_data = {
        "Time": time_value
    }

    input_data.update(
        feature_values
    )

    input_data["Amount"] = amount_value

    selected_transaction = pd.Series(
        input_data
    )


# ============================================================
# SHOW SELECTED TRANSACTION
# ============================================================

if selected_transaction is not None:

    st.subheader(
        "Selected Transaction"
    )

    transaction_preview = pd.DataFrame(
        [selected_transaction]
    )

    st.dataframe(
        transaction_preview,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze = st.button(
    "🔍 Analyze Transaction",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if analyze and selected_transaction is not None:

    # Create dataframe
    input_df = pd.DataFrame(
        [selected_transaction]
    )

    # Make sure column order exactly matches training
    input_df = input_df[
        X_test.columns
    ]

    # Predict probability
    fraud_probability = model.predict_proba(
        input_df
    )[0][1]

    risk_score = fraud_probability * 100


    # ========================================================
    # RISK LEVEL + ACTION
    # ========================================================

    if fraud_probability < threshold:

        risk_level = "LOW"
        action = "ALLOW"

    elif fraud_probability < 0.50:

        risk_level = "MEDIUM"
        action = "REVIEW"

    else:

        risk_level = "HIGH"
        action = "REVIEW"


    # ========================================================
    # RESULT
    # ========================================================

    st.header(
        "📊 Risk Assessment Result"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Risk Score",
            f"{risk_score:.2f}/100"
        )

    with col3:

        st.metric(
            "Risk Level",
            risk_level
        )

    with col4:

        st.metric(
            "Recommended Action",
            action
        )


    # ========================================================
    # DECISION MESSAGE
    # ========================================================

    if action == "ALLOW":

        st.success(
            "✅ LOW RISK — Transaction can be allowed."
        )

    else:

        st.warning(
            "⚠️ POTENTIAL FRAUD — Transaction should be sent for manual review."
        )


    # ========================================================
    # ACTUAL LABEL FOR DEMO SAMPLES
    # ========================================================

    if actual_class is not None:

        if actual_class == 1:

            st.write(
                "Dataset label: **Fraudulent transaction**"
            )

        else:

            st.write(
                "Dataset label: **Legitimate transaction**"
            )


    # ========================================================
    # MODEL EXPLANATION
    # ========================================================

    st.subheader(
        "🔎 Model Explanation"
    )

    st.write(
        """
        These are the features with the highest overall
        importance in the Random Forest model.
        """
    )

    top_features = feature_importance.head(5)

    explanation_data = []

    for _, row in top_features.iterrows():

        feature = row["Feature"]

        explanation_data.append(
            {
                "Feature": feature,
                "Transaction Value": input_df.iloc[0][feature],
                "Global Model Importance": round(
                    row["Importance"],
                    6
                )
            }
        )

    explanation_df = pd.DataFrame(
        explanation_data
    )

    st.dataframe(
        explanation_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Global feature importance shows which features are influential "
        "to the model overall; it does not imply causation."
    )


# ============================================================
# FINAL MODEL PERFORMANCE
# ============================================================

st.divider()

st.header(
    "📈 Final Held-Out Test Performance"
)

metric_dict = dict(
    zip(
        final_metrics["Metric"],
        final_metrics["Value"]
    )
)


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Precision",
        f"{metric_dict['Precision'] * 100:.2f}%"
    )

with col2:

    st.metric(
        "Recall",
        f"{metric_dict['Recall'] * 100:.2f}%"
    )

with col3:

    st.metric(
        "F1 Score",
        f"{metric_dict['F1 Score'] * 100:.2f}%"
    )

with col4:

    st.metric(
        "PR-AUC",
        f"{metric_dict['PR-AUC'] * 100:.2f}%"
    )


st.write(
    f"""
    **Final test set:** 56,746 transactions, including 74 fraud cases.

    **False Positive Rate:** {metric_dict['False Positive Rate'] * 100:.4f}%

    **True Positives:** {int(metric_dict['True Positives'])}

    **False Positives:** {int(metric_dict['False Positives'])}

    **False Negatives:** {int(metric_dict['False Negatives'])}

    **True Negatives:** {int(metric_dict['True Negatives'])}
    """
)


# ============================================================
# COST ANALYSIS
# ============================================================

st.subheader(
    "💰 Cost-Aware Decision"
)

st.write(
    f"""
    The selected decision threshold is **{threshold:.2f}**.

    Project-level assumptions:

    - False Positive Cost = 1
    - False Negative Cost = 10
    - Final estimated cost = **{int(metric_dict['Estimated Cost'])}**

    These costs are modeling assumptions for this project and
    do not represent Razorpay's internal costs.
    """
)


# ============================================================
# ABOUT
# ============================================================

st.divider()

st.header(
    "ℹ️ About the System"
)

st.write(
    """
    This system compares multiple machine learning approaches
    and uses a Random Forest classifier for fraud risk detection.

    The decision threshold was selected using a cost-sensitive
    validation analysis.

    The final performance was measured on a completely held-out
    test set that was not used during model training or threshold
    selection.

    The system is designed strictly for defensive fraud detection
    and manual review support.
    """
)