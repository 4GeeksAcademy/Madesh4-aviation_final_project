import streamlit as st
import pandas as pd
import numpy as np
from model import Model
from utils import load_data, plot_feature_distribution, plot_correlation_matrix, plot_confusion_matrix, get_classification_metrics

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = Model()
    st.session_state.model.load_model()

# Title and description
st.title("💳 Credit Card Fraud Detection")
st.markdown("""
This application helps detect fraudulent credit card transactions using machine learning.
Upload a CSV file with transaction data or use the form below to make predictions.
""")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Data Analysis"])

if page == "Prediction":
    st.header("Make Predictions")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(data.head())
        
        if st.button("Predict"):
            predictions = []
            for _, row in data.iterrows():
                pred = st.session_state.model.predict(row.to_dict())
                predictions.append(pred)
            
            results_df = pd.DataFrame(predictions)
            st.write("Prediction Results:")
            st.dataframe(results_df)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
    
    # Manual input form
    st.subheader("Or enter transaction details manually")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time = st.number_input("Time", min_value=0.0)
        amount = st.number_input("Amount", min_value=0.0)
        v1 = st.number_input("V1", value=0.0)
        v2 = st.number_input("V2", value=0.0)
        v3 = st.number_input("V3", value=0.0)
        v4 = st.number_input("V4", value=0.0)
        v5 = st.number_input("V5", value=0.0)
    
    with col2:
        v6 = st.number_input("V6", value=0.0)
        v7 = st.number_input("V7", value=0.0)
        v8 = st.number_input("V8", value=0.0)
        v9 = st.number_input("V9", value=0.0)
        v10 = st.number_input("V10", value=0.0)
        v11 = st.number_input("V11", value=0.0)
        v12 = st.number_input("V12", value=0.0)
    
    if st.button("Predict Transaction"):
        features = {
            'Time': time,
            'Amount': amount,
            'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V5': v5,
            'V6': v6, 'V7': v7, 'V8': v8, 'V9': v9, 'V10': v10,
            'V11': v11, 'V12': v12
        }
        
        prediction = st.session_state.model.predict(features)
        
        st.subheader("Prediction Result")
        if prediction['prediction'] == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")
        
        st.write(f"Confidence: {prediction['probability']:.2%}")
        
        # Show class probabilities
        st.write("Class Probabilities:")
        for class_name, prob in prediction['class_probabilities'].items():
            st.write(f"{'Fraudulent' if class_name == 1 else 'Legitimate'}: {prob:.2%}")

else:  # Data Analysis page
    st.header("Data Analysis")
    
    # Load sample data
    data = load_data('data/creditcard.csv')
    
    if data is not None:
        # Data overview
        st.subheader("Data Overview")
        st.write(f"Number of transactions: {len(data)}")
        st.write(f"Number of features: {len(data.columns)}")
        st.write(f"Fraud rate: {(data['Class'].mean() * 100):.2f}%")
        
        # Feature distributions
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select feature to visualize", data.columns)
        fig = plot_feature_distribution(data, feature, 'Class')
        st.plotly_chart(fig)
        
        # Correlation matrix
        st.subheader("Feature Correlations")
        corr_fig = plot_correlation_matrix(data)
        st.plotly_chart(corr_fig)
        
        # Download sample data
        st.download_button(
            label="Download sample data",
            data=data.head(1000).to_csv(index=False),
            file_name="sample_data.csv",
            mime="text/csv"
        )