# Flight Incident Predictor

A Streamlit web application that predicts the probability of flight incidents based on various flight parameters. The application uses machine learning to analyze historical flight data and provide risk assessments for future flights.

## Features

- **Incident Prediction**: Predict the probability of incidents for specific flights based on:
  - Origin and destination airports
  - Date and time of flight
  - Aircraft type
  - Weather conditions
  - Flight duration

- **Model Performance Analysis**: View detailed metrics about the model's performance:
  - Accuracy, precision, and recall metrics
  - Confusion matrix visualization
  - Feature importance analysis

- **Data Exploration**: Interactive visualizations of flight incident data:
  - Time series analysis of incidents
  - Airport-specific incident statistics
  - Weather impact analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd flight-incident-predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

To use `parse_access` in Python, install the `mdbtools` package in your Codespace:
```bash
sudo apt update
sudo apt upgrade
sudo apt install mdbtools
```

## Project Structure

```
flight-incident-predictor/
├── app.py                 # Main Streamlit application
├── utils.py              # Utility functions
├── requirements.txt      # Project dependencies
├── models/              # Directory for trained models
│   └── flight_incident_model.joblib
└── data/               # Data directory
    ├── airports.csv
    └── weather_data.csv
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the application:
   - Select the "Incident Prediction" tab to make predictions
   - Fill in the flight details
   - Click "Predict Incident Probability" to see the results
   - Explore model performance and data analysis in other tabs

## Model Details

The application uses a HistGradientBoostingClassifier with the following features:
- Time-based features (hour, day of week, month)
- Flight duration
- Origin and destination airports
- Aircraft type
- Weather conditions

Model performance metrics:
- Accuracy: 99.7%
- Precision: 99.8%
- Recall: 99.6%

## Data Sources

The model is trained on historical flight data including:
- Flight schedules and delays
- Weather conditions
- Aircraft information
- Airport statistics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data sources and providers
- Contributors and maintainers
- Open-source community

## Contact

For questions or support, please open an issue in the repository.