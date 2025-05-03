<<<<<<< HEAD
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
=======
import configuration as config 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.title("Worried Your Plane might Crash?")
st.subheader("Submit your flight details to find out.")
st.divider()

st.image(os.path.join(os.path.dirname(__file__), "static", "photo.jpg"))

# load dataset
data_path = os.path.join(os.getcwd(), "data", "processed", "combined_data.csv")
data_df = pd.read_csv(data_path)

flight_details = {
        "origin": None,
        "destination": None,
        "departure_time": None,
    }

with st.form(key="user_flight_details"):
    flight_details["origin"] = st.selectbox(label="Enter the departure airport: ", options=data_df['origin'].unique(), placeholder='LGA')
    flight_details["destination"] = st.selectbox(label="Enter the destination airport: ", options=data_df['destination'].unique(), placeholder='ORF')
    flight_details["departure_time"] = st.time_input("Enter your departure time (use military time): ")

    submit = st.form_submit_button("Submit")

if submit:
    if not all(flight_details.values()):
        st.warning("Please fill in all of the fields")
    else:
        # Convert departure_time (which is a time object) to string
        df = pd.DataFrame({
            "origin": [flight_details["origin"]],
            "destination": [flight_details["destination"]],
            "departure_time": [flight_details["departure_time"].strftime("%H:%M")]
        })
        # create and encode route
        route_frequency = data_df['origin'] + '_' + data_df['destination']
        route_frequency = route_frequency.value_counts().to_dict()
        df['route'] = df['origin'] + '_' + df['destination']
        df['route_encoded'] = df['route'].map(route_frequency)
        df['route_encoded'].fillna(0, inplace=True)
        df.drop(columns=['route'], inplace=True)
        print(df)

        # create and encode time-sin and time-cosine
        def hhmm_to_minutes(hhmm):
            hours, minutes = map(int, hhmm.split(":"))
            return hours * 60 + minutes  
        
        df['Time'] = df['departure_time'].apply(hhmm_to_minutes)
        df['time_sin'] = np.sin(2 * np.pi * df['Time'] / 1440)  # 1440 minutes in a day
        df['time_cos'] = np.cos(2 * np.pi * df['Time'] / 1440)
        
        df = df[['time_sin', 'time_cos', 'route_encoded']]
        
        # Load the trained model
        model_path = os.path.join(os.getcwd(), "models", "model.pkl")


        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.stop()

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # make the predictions
        probability = model.predict_proba(df)
        percent_probability = probability[:, 1] * 100
        print(percent_probability)

        # Display predictions
        st.write(f"The probability of your plane crashing is {percent_probability.item():.2f}%")
>>>>>>> pull_request
