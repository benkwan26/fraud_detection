import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")

model = load_model()

# -----------------------
# Custom Styling
# -----------------------
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3, h4 {
        color: white;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50, #2E7D32);
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .metric-box {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.title("💳 Fraud Detection Dashboard")
st.caption("Analyze financial transactions in real time")

# -----------------------
# Layout
# -----------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Transaction Input")

    transaction_type = st.selectbox(
        "Transaction Type",
        ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT']
    )

    amount = st.number_input('Amount', min_value=0.0, value=1000.0)

    oldbalanceOrg = st.number_input(
        "Old Balance (Sender)", min_value=0.0, value=10000.0
    )
    newbalanceOrig = st.number_input(
        "New Balance (Sender)", min_value=0.0, value=9000.0
    )

    oldbalanceDest = st.number_input(
        "Old Balance (Receiver)", min_value=0.0, value=0.0
    )
    newbalanceDest = st.number_input(
        "New Balance (Receiver)", min_value=0.0, value=0.0
    )

    predict_btn = st.button("Analyze Transaction")

# -----------------------
# Derived Features (Optional Insight)
# -----------------------
balance_diff_orig = oldbalanceOrg - newbalanceOrig
balance_diff_dest = newbalanceDest - oldbalanceDest

# -----------------------
# Right Panel
# -----------------------
with col2:
    st.subheader("Transaction Insights")

    st.markdown("**Quick Metrics**")
    m1, m2 = st.columns(2)
    m1.metric("Sender Balance Change", f"{balance_diff_orig:,.2f}")
    m2.metric("Receiver Balance Change", f"{balance_diff_dest:,.2f}")

    st.markdown("---")

    if predict_btn:
        input_data = pd.DataFrame([{
            'type': transaction_type,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
        }])

        prediction = model.predict(input_data)[0]

        # Try probability if model supports it
        try:
            prob = model.predict_proba(input_data)[0][1]
        except:
            prob = None

        st.subheader("Result")

        if prediction == 1:
            st.error("⚠️ High Risk: Potential Fraud Detected")
        else:
            st.success("✅ Low Risk: Transaction Looks Safe")

        # Probability display
        if prob is not None:
            st.progress(float(prob))
            st.caption(f"Fraud Probability: {prob:.2%}")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("Built with Streamlit • Model: fraud_detection_pipeline.pkl")