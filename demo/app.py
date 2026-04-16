"""Streamlit demo for Identity Theft Detection.

This module provides an interactive web interface for testing and
demonstrating the identity theft detection system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, Any
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.synthetic_data import generate_synthetic_data, IdentityTheftDataGenerator
from src.models.identity_detector import IdentityTheftDetector
from src.eval.evaluator import IdentityTheftEvaluator


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Identity Theft Detection Demo",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🛡️ Identity Theft Detection System")
    st.markdown("**Defensive Research Demonstration - Educational Use Only**")
    
    # Disclaimer
    with st.expander("⚠️ Important Disclaimer", expanded=True):
        st.warning("""
        **This is a defensive research demonstration for educational purposes only.**
        
        - Results may be inaccurate and should not be used for production security operations
        - This is NOT a SOC (Security Operations Center) tool
        - All data is synthetic and de-identified for demonstration purposes
        - No offensive capabilities or exploitation techniques are implemented
        """)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["xgboost", "random_forest", "lightgbm", "neural_net"],
        help="Choose the machine learning model for identity theft detection"
    )
    
    # Dataset size
    dataset_size = st.sidebar.slider(
        "Dataset Size",
        min_value=1000,
        max_value=10000,
        value=5000,
        step=1000,
        help="Number of synthetic samples to generate"
    )
    
    # Threshold
    threshold = st.sidebar.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probability threshold for fraud classification"
    )
    
    # Generate data button
    if st.sidebar.button("🔄 Generate New Dataset"):
        st.session_state.data_generated = False
    
    # Main content
    if 'data_generated' not in st.session_state or not st.session_state.data_generated:
        generate_and_train_model(model_type, dataset_size, threshold)
    else:
        display_results()


def generate_and_train_model(model_type: str, dataset_size: int, threshold: float):
    """Generate data and train model."""
    st.header("📊 Data Generation & Model Training")
    
    with st.spinner("Generating synthetic dataset..."):
        # Generate synthetic data
        data = generate_synthetic_data(dataset_size)
        
        # Store in session state
        st.session_state.data = data
        st.session_state.model_type = model_type
        st.session_state.threshold = threshold
        st.session_state.data_generated = True
    
    st.success(f"✅ Generated {dataset_size:,} synthetic samples")
    
    # Display dataset info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{dataset_size:,}")
    
    with col2:
        fraud_rate = data['metadata']['fraud_rate']
        st.metric("Fraud Rate", f"{fraud_rate:.1%}")
    
    with col3:
        n_features = data['metadata']['n_features']
        st.metric("Features", n_features)
    
    with col4:
        train_samples = len(data['X_train'])
        st.metric("Training Samples", f"{train_samples:,}")
    
    # Train model
    with st.spinner(f"Training {model_type} model..."):
        detector = IdentityTheftDetector(model_type=model_type)
        detector.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
        
        # Store model
        st.session_state.detector = detector
    
    st.success(f"✅ {model_type.upper()} model trained successfully")
    
    # Make predictions
    with st.spinner("Making predictions..."):
        y_pred = detector.predict(data['X_test'])
        y_proba = detector.predict_proba(data['X_test'])[:, 1]
        
        # Store predictions
        st.session_state.y_pred = y_pred
        st.session_state.y_proba = y_proba
    
    st.success("✅ Predictions completed")
    
    # Evaluate model
    with st.spinner("Evaluating model performance..."):
        evaluator = IdentityTheftEvaluator(model_type.upper())
        metrics = evaluator.evaluate(data['y_test'], y_pred, y_proba)
        
        # Store evaluator
        st.session_state.evaluator = evaluator
        st.session_state.metrics = metrics
    
    st.success("✅ Model evaluation completed")
    
    # Display key metrics
    st.subheader("🎯 Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
    
    with col2:
        st.metric("AUC-PR", f"{metrics['auc_pr']:.3f}")
    
    with col3:
        st.metric("Precision@5%", f"{metrics['precision_at_5']:.3f}")
    
    with col4:
        st.metric("Alert Efficiency", f"{metrics['alert_efficiency']:.1%}")


def display_results():
    """Display model results and analysis."""
    data = st.session_state.data
    detector = st.session_state.detector
    evaluator = st.session_state.evaluator
    metrics = st.session_state.metrics
    threshold = st.session_state.threshold
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance Metrics", 
        "🔍 Feature Analysis", 
        "📊 Visualizations", 
        "🚨 Alert Analysis",
        "📋 Detailed Report"
    ])
    
    with tab1:
        display_performance_metrics(metrics)
    
    with tab2:
        display_feature_analysis(detector)
    
    with tab3:
        display_visualizations(data, evaluator)
    
    with tab4:
        display_alert_analysis(data, threshold)
    
    with tab5:
        display_detailed_report(data, evaluator)


def display_performance_metrics(metrics: Dict[str, float]):
    """Display performance metrics."""
    st.subheader("📈 Model Performance Metrics")
    
    # Basic metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Metrics")
        basic_metrics = {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'AUC-ROC': metrics['auc_roc'],
            'AUC-PR': metrics['auc_pr']
        }
        
        for metric, value in basic_metrics.items():
            st.metric(metric, f"{value:.4f}")
    
    with col2:
        st.subheader("Security Metrics")
        security_metrics = {
            'Precision@1%': metrics['precision_at_1'],
            'Precision@5%': metrics['precision_at_5'],
            'Precision@10%': metrics['precision_at_10'],
            'Precision@20%': metrics['precision_at_20']
        }
        
        for metric, value in security_metrics.items():
            st.metric(metric, f"{value:.4f}")
    
    # Alert metrics
    st.subheader("🚨 Alert Workload Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Alert Rate", f"{metrics['alert_rate']:.2%}")
    
    with col2:
        st.metric("Alerts per 1000", f"{metrics['alerts_per_1000']:.1f}")
    
    with col3:
        st.metric("Alert Efficiency", f"{metrics['alert_efficiency']:.1%}")
    
    with col4:
        st.metric("False Alerts per 1000", f"{metrics['false_alerts_per_1000']:.1f}")


def display_feature_analysis(detector: IdentityTheftDetector):
    """Display feature importance analysis."""
    st.subheader("🔍 Feature Importance Analysis")
    
    # Get feature importance
    importance = detector.get_feature_importance()
    
    if importance.sum() > 0:
        # Top features
        top_features = importance.head(10)
        
        # Create bar chart
        fig = px.bar(
            x=top_features.values,
            y=top_features.index,
            orientation='h',
            title="Top 10 Most Important Features",
            labels={'x': 'Importance Score', 'y': 'Feature'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature table
        st.subheader("Feature Importance Scores")
        importance_df = pd.DataFrame({
            'Feature': importance.index,
            'Importance': importance.values
        }).sort_values('Importance', ascending=False)
        
        st.dataframe(importance_df, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")


def display_visualizations(data: Dict[str, Any], evaluator: IdentityTheftEvaluator):
    """Display model visualizations."""
    st.subheader("📊 Model Visualizations")
    
    # Get test data
    y_test = data['y_test']
    y_proba = st.session_state.y_proba
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('ROC Curve', 'Precision-Recall Curve', 
                       'Calibration Curve', 'Confusion Matrix'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # ROC Curve
    fpr, tpr, _ = evaluator._calculate_security_metrics(y_test, y_test, y_proba)
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'),
        row=1, col=1
    )
    
    # Precision-Recall Curve
    precision, recall, _ = evaluator._calculate_security_metrics(y_test, y_test, y_proba)
    fig.add_trace(
        go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve'),
        row=1, col=2
    )
    
    # Calibration Curve
    fraction_of_positives, mean_predicted_value = evaluator._calculate_calibration_metrics(y_test, y_proba)
    fig.add_trace(
        go.Scatter(x=mean_predicted_value, y=fraction_of_positives, 
                  mode='lines+markers', name='Calibration'),
        row=2, col=1
    )
    
    # Confusion Matrix
    y_pred = st.session_state.y_pred
    cm = evaluator._calculate_alert_metrics(y_test, y_pred, y_proba)
    fig.add_trace(
        go.Heatmap(z=cm, name='Confusion Matrix'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def display_alert_analysis(data: Dict[str, Any], threshold: float):
    """Display alert analysis."""
    st.subheader("🚨 Alert Analysis")
    
    # Get predictions with threshold
    y_proba = st.session_state.y_proba
    y_pred_thresh = (y_proba >= threshold).astype(int)
    
    # Alert statistics
    n_alerts = y_pred_thresh.sum()
    n_total = len(y_pred_thresh)
    n_fraud = data['y_test'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", n_alerts)
    
    with col2:
        st.metric("Alert Rate", f"{n_alerts/n_total:.2%}")
    
    with col3:
        true_alerts = (y_pred_thresh & data['y_test']).sum()
        st.metric("True Alerts", true_alerts)
    
    with col4:
        false_alerts = (y_pred_thresh & ~data['y_test']).sum()
        st.metric("False Alerts", false_alerts)
    
    # Alert distribution
    st.subheader("Alert Distribution by Risk Score")
    
    # Create histogram
    fig = px.histogram(
        x=y_proba,
        color=data['y_test'].astype(str),
        title="Distribution of Risk Scores",
        labels={'x': 'Risk Score', 'y': 'Count'},
        color_discrete_map={'0': 'blue', '1': 'red'}
    )
    
    # Add threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold:.2f}"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # High-risk cases
    st.subheader("🔴 High-Risk Cases")
    
    high_risk_indices = np.where(y_proba >= threshold)[0]
    
    if len(high_risk_indices) > 0:
        high_risk_df = pd.DataFrame({
            'Index': high_risk_indices,
            'Risk Score': y_proba[high_risk_indices],
            'Actual Fraud': data['y_test'].iloc[high_risk_indices],
            'Predicted Fraud': y_pred_thresh[high_risk_indices]
        }).sort_values('Risk Score', ascending=False)
        
        st.dataframe(high_risk_df.head(20), use_container_width=True)
    else:
        st.info("No high-risk cases found with current threshold.")


def display_detailed_report(data: Dict[str, Any], evaluator: IdentityTheftEvaluator):
    """Display detailed evaluation report."""
    st.subheader("📋 Detailed Evaluation Report")
    
    # Generate report
    y_test = data['y_test']
    y_pred = st.session_state.y_pred
    y_proba = st.session_state.y_proba
    
    report = evaluator.generate_report(y_test, y_pred, y_proba)
    
    # Display report
    st.text(report)
    
    # Download button
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="identity_theft_evaluation_report.txt",
        mime="text/plain"
    )


if __name__ == "__main__":
    main()
