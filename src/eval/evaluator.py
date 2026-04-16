"""Evaluation module for identity theft detection.

This module provides comprehensive evaluation metrics and analysis tools
specifically designed for identity theft detection systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class IdentityTheftEvaluator:
    """Comprehensive evaluator for identity theft detection models."""
    
    def __init__(self, model_name: str = "Model") -> None:
        """Initialize the evaluator.
        
        Args:
            model_name: Name of the model being evaluated.
        """
        self.model_name = model_name
        self.results = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities.
            threshold: Classification threshold.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Basic classification metrics
        metrics = {
            'accuracy': (y_pred == y_true).mean(),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_proba),
            'auc_pr': average_precision_score(y_true, y_proba)
        }
        
        # Security-relevant metrics
        metrics.update(self._calculate_security_metrics(y_true, y_pred, y_proba))
        
        # Alert workload metrics
        metrics.update(self._calculate_alert_metrics(y_true, y_pred, y_proba))
        
        # Calibration metrics
        metrics.update(self._calculate_calibration_metrics(y_true, y_proba))
        
        self.results = metrics
        return metrics
    
    def _calculate_security_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate security-specific metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities.
            
        Returns:
            Dictionary of security metrics.
        """
        metrics = {}
        
        # Precision@K metrics (critical for fraud detection)
        for k in [1, 5, 10, 20]:
            k_percent = k / 100
            n_top_k = max(1, int(len(y_true) * k_percent))
            top_k_indices = np.argsort(y_proba)[-n_top_k:]
            precision_at_k = y_true[top_k_indices].mean()
            metrics[f'precision_at_{k}'] = precision_at_k
        
        # Recall at fixed precision levels
        precision_levels = [0.5, 0.7, 0.9]
        for prec_level in precision_levels:
            recall_at_precision = self._recall_at_precision(y_true, y_proba, prec_level)
            metrics[f'recall_at_precision_{int(prec_level*100)}'] = recall_at_precision
        
        # False Positive Rate at different TPR levels
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        tpr_levels = [0.5, 0.7, 0.9]
        for tpr_level in tpr_levels:
            fpr_at_tpr = self._fpr_at_tpr(fpr, tpr, tpr_level)
            metrics[f'fpr_at_tpr_{int(tpr_level*100)}'] = fpr_at_tpr
        
        return metrics
    
    def _calculate_alert_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate alert workload metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities.
            
        Returns:
            Dictionary of alert metrics.
        """
        metrics = {}
        
        # Alert volume metrics
        n_alerts = y_pred.sum()
        n_total = len(y_pred)
        n_fraud = y_true.sum()
        
        metrics['alert_rate'] = n_alerts / n_total
        metrics['alerts_per_1000'] = (n_alerts / n_total) * 1000
        metrics['alerts_per_fraud_case'] = n_alerts / max(n_fraud, 1)
        
        # Alert efficiency
        true_alerts = (y_pred & y_true).sum()
        false_alerts = (y_pred & ~y_true).sum()
        
        metrics['alert_efficiency'] = true_alerts / max(n_alerts, 1)
        metrics['false_alerts_per_1000'] = (false_alerts / n_total) * 1000
        
        return metrics
    
    def _calculate_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate calibration metrics.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            
        Returns:
            Dictionary of calibration metrics.
        """
        metrics = {}
        
        # Expected Calibration Error (ECE)
        ece = self._expected_calibration_error(y_true, y_proba)
        metrics['expected_calibration_error'] = ece
        
        # Maximum Calibration Error (MCE)
        mce = self._maximum_calibration_error(y_true, y_proba)
        metrics['maximum_calibration_error'] = mce
        
        return metrics
    
    def _recall_at_precision(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        target_precision: float
    ) -> float:
        """Calculate recall at a specific precision level.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            target_precision: Target precision level.
            
        Returns:
            Recall at target precision.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Find threshold that achieves target precision
        valid_indices = precision >= target_precision
        if not valid_indices.any():
            return 0.0
        
        best_idx = np.argmax(recall[valid_indices])
        return recall[valid_indices][best_idx]
    
    def _fpr_at_tpr(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        target_tpr: float
    ) -> float:
        """Calculate FPR at a specific TPR level.
        
        Args:
            fpr: False positive rates.
            tpr: True positive rates.
            target_tpr: Target TPR level.
            
        Returns:
            FPR at target TPR.
        """
        # Find closest TPR to target
        idx = np.argmin(np.abs(tpr - target_tpr))
        return fpr[idx]
    
    def _expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            n_bins: Number of bins for calibration.
            
        Returns:
            Expected Calibration Error.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _maximum_calibration_error(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Calculate Maximum Calibration Error.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            n_bins: Number of bins for calibration.
            
        Returns:
            Maximum Calibration Error.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> go.Figure:
        """Plot ROC curve.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            
        Returns:
            Plotly figure with ROC curve.
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{self.model_name} (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500
        )
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> go.Figure:
        """Plot Precision-Recall curve.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            
        Returns:
            Plotly figure with PR curve.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auc_pr = average_precision_score(y_true, y_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'{self.model_name} (AUC-PR = {auc_pr:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Add baseline
        baseline = y_true.mean()
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Baseline (AP = {baseline:.3f})"
        )
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=600,
            height=500
        )
        
        return fig
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> go.Figure:
        """Plot calibration curve.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            
        Returns:
            Plotly figure with calibration curve.
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10
        )
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode='lines+markers',
            name=f'{self.model_name}',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # Add perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Calibration Curve',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            width=600,
            height=500
        )
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            Plotly figure with confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Normal', 'Predicted Fraud'],
            y=['Actual Normal', 'Actual Fraud'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={'size': 16}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            width=500,
            height=400
        )
        
        return fig
    
    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities.
            save_path: Path to save the report.
            
        Returns:
            Report text.
        """
        # Calculate metrics
        metrics = self.evaluate(y_true, y_pred, y_proba)
        
        # Generate report
        report = f"""
Identity Theft Detection Model Evaluation Report
==============================================

Model: {self.model_name}
Dataset Size: {len(y_true):,} samples
Fraud Rate: {y_true.mean():.2%}

Performance Metrics:
--------------------
Accuracy: {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1-Score: {metrics['f1']:.4f}
AUC-ROC: {metrics['auc_roc']:.4f}
AUC-PR: {metrics['auc_pr']:.4f}

Security Metrics:
-----------------
Precision@1%: {metrics['precision_at_1']:.4f}
Precision@5%: {metrics['precision_at_5']:.4f}
Precision@10%: {metrics['precision_at_10']:.4f}
Precision@20%: {metrics['precision_at_20']:.4f}

Recall at Precision 50%: {metrics['recall_at_precision_50']:.4f}
Recall at Precision 70%: {metrics['recall_at_precision_70']:.4f}
Recall at Precision 90%: {metrics['recall_at_precision_90']:.4f}

FPR at TPR 50%: {metrics['fpr_at_tpr_50']:.4f}
FPR at TPR 70%: {metrics['fpr_at_tpr_70']:.4f}
FPR at TPR 90%: {metrics['fpr_at_tpr_90']:.4f}

Alert Workload Metrics:
----------------------
Alert Rate: {metrics['alert_rate']:.2%}
Alerts per 1000 Events: {metrics['alerts_per_1000']:.1f}
Alerts per Fraud Case: {metrics['alerts_per_fraud_case']:.1f}
Alert Efficiency: {metrics['alert_efficiency']:.2%}
False Alerts per 1000: {metrics['false_alerts_per_1000']:.1f}

Calibration Metrics:
-------------------
Expected Calibration Error: {metrics['expected_calibration_error']:.4f}
Maximum Calibration Error: {metrics['maximum_calibration_error']:.4f}

Recommendations:
---------------
"""
        
        # Add recommendations based on metrics
        if metrics['alert_rate'] > 0.1:
            report += "- High alert rate detected. Consider increasing threshold to reduce false positives.\n"
        
        if metrics['alert_efficiency'] < 0.5:
            report += "- Low alert efficiency. Model may need retraining or feature engineering.\n"
        
        if metrics['expected_calibration_error'] > 0.1:
            report += "- Poor calibration detected. Consider using calibrated probabilities.\n"
        
        if metrics['precision_at_5'] < 0.3:
            report += "- Low precision at top 5%. Consider ensemble methods or additional features.\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report


def compare_models(
    results: Dict[str, Dict[str, float]],
    metric: str = 'auc_pr'
) -> pd.DataFrame:
    """Compare multiple models on a specific metric.
    
    Args:
        results: Dictionary of model results.
        metric: Metric to compare on.
        
    Returns:
        DataFrame with model comparison.
    """
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Metric': metric,
            'Value': metrics.get(metric, 0.0)
        })
    
    df = pd.DataFrame(comparison_data)
    return df.sort_values('Value', ascending=False)


if __name__ == "__main__":
    # Test the evaluator
    from src.data.synthetic_data import generate_synthetic_data
    from src.models.identity_detector import IdentityTheftDetector
    
    # Generate data
    data = generate_synthetic_data(1000)
    
    # Train model
    detector = IdentityTheftDetector(model_type="xgboost")
    detector.fit(data['X_train'], data['y_train'])
    
    # Make predictions
    y_pred = detector.predict(data['X_test'])
    y_proba = detector.predict_proba(data['X_test'])[:, 1]
    
    # Evaluate
    evaluator = IdentityTheftEvaluator("XGBoost")
    metrics = evaluator.evaluate(data['y_test'], y_pred, y_proba)
    
    print("Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate report
    report = evaluator.generate_report(data['y_test'], y_pred, y_proba)
    print(report)
