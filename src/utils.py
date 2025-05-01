import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    """Load data from CSV file"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def plot_feature_distribution(data, feature, target):
    """Plot feature distribution by target class"""
    fig = px.histogram(data, x=feature, color=target,
                      title=f'Distribution of {feature} by {target}',
                      barmode='overlay')
    return fig

def plot_correlation_matrix(data):
    """Plot correlation matrix heatmap"""
    corr_matrix = data.corr()
    fig = px.imshow(corr_matrix,
                    title='Feature Correlation Matrix',
                    color_continuous_scale='RdBu')
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Negative', 'Positive'],
                    y=['Negative', 'Positive'],
                    title='Confusion Matrix')
    return fig

def get_classification_metrics(y_true, y_pred):
    """Get classification metrics"""
    report = classification_report(y_true, y_pred, output_dict=True)
    return report