"""Explainability module for identity theft detection.

This module provides SHAP-based explanations and rule-based evidence
for identity theft detection models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


class IdentityTheftExplainer:
    """Explainability engine for identity theft detection models."""
    
    def __init__(self, model, feature_names: List[str]) -> None:
        """Initialize the explainer.
        
        Args:
            model: Trained identity theft detection model.
            feature_names: List of feature names.
        """
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = None
        self.shap_values = None
        
        # Initialize SHAP explainer if available
        if SHAP_AVAILABLE:
            self._initialize_shap_explainer()
    
    def _initialize_shap_explainer(self) -> None:
        """Initialize SHAP explainer based on model type."""
        if not SHAP_AVAILABLE:
            return
        
        try:
            if hasattr(self.model, 'predict_proba'):
                # For tree-based models
                if hasattr(self.model, 'feature_importances_'):
                    self.shap_explainer = shap.TreeExplainer(self.model)
                else:
                    # For other models, use KernelExplainer
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        self.model.X_train if hasattr(self.model, 'X_train') else None
                    )
        except Exception as e:
            warnings.warn(f"Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        instance_idx: int = 0,
        method: str = "shap"
    ) -> Dict[str, Any]:
        """Explain a single prediction.
        
        Args:
            X: Feature matrix.
            instance_idx: Index of instance to explain.
            method: Explanation method ('shap', 'permutation').
            
        Returns:
            Dictionary with explanation results.
        """
        instance = X.iloc[instance_idx:instance_idx+1]
        
        if method == "shap" and self.shap_explainer is not None:
            return self._explain_with_shap(instance)
        else:
            return self._explain_with_permutation(instance)
    
    def _explain_with_shap(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """Explain using SHAP values.
        
        Args:
            instance: Single instance to explain.
            
        Returns:
            SHAP explanation results.
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return {"error": "SHAP not available"}
        
        try:
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(instance)
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            # Get prediction
            prediction = self.model.predict_proba(instance)[0]
            
            # Create explanation
            explanation = {
                'prediction': prediction[1],  # Fraud probability
                'shap_values': shap_values[0],
                'feature_names': self.feature_names,
                'base_value': self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, list) else self.shap_explainer.expected_value,
                'method': 'shap'
            }
            
            return explanation
            
        except Exception as e:
            return {"error": f"SHAP explanation failed: {e}"}
    
    def _explain_with_permutation(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """Explain using permutation importance.
        
        Args:
            instance: Single instance to explain.
            
        Returns:
            Permutation importance explanation.
        """
        try:
            # Get prediction
            prediction = self.model.predict_proba(instance)[0]
            
            # Calculate feature importance by permutation
            importance_scores = []
            
            for i, feature in enumerate(self.feature_names):
                # Create perturbed instance
                perturbed_instance = instance.copy()
                perturbed_instance.iloc[0, i] = np.random.permutation(instance.iloc[0, i:i+1])[0]
                
                # Get perturbed prediction
                perturbed_prediction = self.model.predict_proba(perturbed_instance)[0]
                
                # Calculate importance as difference in prediction
                importance = abs(prediction[1] - perturbed_prediction[1])
                importance_scores.append(importance)
            
            explanation = {
                'prediction': prediction[1],
                'importance_scores': importance_scores,
                'feature_names': self.feature_names,
                'method': 'permutation'
            }
            
            return explanation
            
        except Exception as e:
            return {"error": f"Permutation explanation failed: {e}"}
    
    def explain_batch(
        self,
        X: pd.DataFrame,
        n_samples: int = 100,
        method: str = "shap"
    ) -> Dict[str, Any]:
        """Explain a batch of predictions.
        
        Args:
            X: Feature matrix.
            n_samples: Number of samples to explain.
            method: Explanation method.
            
        Returns:
            Batch explanation results.
        """
        # Sample instances
        if len(X) > n_samples:
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X.iloc[sample_indices]
        else:
            X_sample = X
            sample_indices = np.arange(len(X))
        
        explanations = []
        
        for i in range(len(X_sample)):
            explanation = self.explain_prediction(X_sample, i, method)
            explanations.append(explanation)
        
        # Aggregate explanations
        if method == "shap" and all('shap_values' in exp for exp in explanations):
            # Average SHAP values
            avg_shap_values = np.mean([exp['shap_values'] for exp in explanations], axis=0)
            
            return {
                'method': 'shap',
                'avg_shap_values': avg_shap_values,
                'feature_names': self.feature_names,
                'n_samples': len(X_sample),
                'individual_explanations': explanations
            }
        else:
            # Average importance scores
            avg_importance = np.mean([exp['importance_scores'] for exp in explanations], axis=0)
            
            return {
                'method': 'permutation',
                'avg_importance_scores': avg_importance,
                'feature_names': self.feature_names,
                'n_samples': len(X_sample),
                'individual_explanations': explanations
            }
    
    def generate_rule_evidence(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Generate rule-based evidence for predictions.
        
        Args:
            X: Feature matrix.
            y: True labels.
            threshold: Classification threshold.
            
        Returns:
            Rule-based evidence.
        """
        # Get predictions
        y_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        # Analyze feature patterns for fraud cases
        fraud_cases = X[y == 1]
        normal_cases = X[y == 0]
        
        rules = []
        
        # Rule 1: Suspicious login hours
        suspicious_hours = [0, 1, 2, 3, 4, 5, 23]  # Late night/early morning
        if 'login_hour' in X.columns:
            fraud_suspicious_hour_rate = (fraud_cases['login_hour'].isin(suspicious_hours)).mean()
            normal_suspicious_hour_rate = (normal_cases['login_hour'].isin(suspicious_hours)).mean()
            
            if fraud_suspicious_hour_rate > normal_suspicious_hour_rate * 2:
                rules.append({
                    'rule': 'Suspicious Login Hours',
                    'description': f'Fraud cases occur {fraud_suspicious_hour_rate:.1%} of the time during suspicious hours vs {normal_suspicious_hour_rate:.1%} for normal cases',
                    'risk_factor': 'High',
                    'feature': 'login_hour'
                })
        
        # Rule 2: Device mismatch
        if 'device_match' in X.columns:
            fraud_device_mismatch_rate = (fraud_cases['device_match'] == 0).mean()
            normal_device_mismatch_rate = (normal_cases['device_match'] == 0).mean()
            
            if fraud_device_mismatch_rate > normal_device_mismatch_rate * 1.5:
                rules.append({
                    'rule': 'Device Mismatch',
                    'description': f'Fraud cases show {fraud_device_mismatch_rate:.1%} device mismatch vs {normal_device_mismatch_rate:.1%} for normal cases',
                    'risk_factor': 'High',
                    'feature': 'device_match'
                })
        
        # Rule 3: Location mismatch
        if 'location_match' in X.columns:
            fraud_location_mismatch_rate = (fraud_cases['location_match'] == 0).mean()
            normal_location_mismatch_rate = (normal_cases['location_match'] == 0).mean()
            
            if fraud_location_mismatch_rate > normal_location_mismatch_rate * 1.5:
                rules.append({
                    'rule': 'Location Mismatch',
                    'description': f'Fraud cases show {fraud_location_mismatch_rate:.1%} location mismatch vs {normal_location_mismatch_rate:.1%} for normal cases',
                    'risk_factor': 'High',
                    'feature': 'location_match'
                })
        
        # Rule 4: Multiple attempts
        if 'multiple_attempts' in X.columns:
            fraud_multiple_attempts_rate = (fraud_cases['multiple_attempts'] == 1).mean()
            normal_multiple_attempts_rate = (normal_cases['multiple_attempts'] == 1).mean()
            
            if fraud_multiple_attempts_rate > normal_multiple_attempts_rate * 2:
                rules.append({
                    'rule': 'Multiple Login Attempts',
                    'description': f'Fraud cases show {fraud_multiple_attempts_rate:.1%} multiple attempts vs {normal_multiple_attempts_rate:.1%} for normal cases',
                    'risk_factor': 'Medium',
                    'feature': 'multiple_attempts'
                })
        
        # Rule 5: High velocity score
        if 'velocity_score' in X.columns:
            fraud_high_velocity_rate = (fraud_cases['velocity_score'] > 0.8).mean()
            normal_high_velocity_rate = (normal_cases['velocity_score'] > 0.8).mean()
            
            if fraud_high_velocity_rate > normal_high_velocity_rate * 2:
                rules.append({
                    'rule': 'High Velocity Score',
                    'description': f'Fraud cases show {fraud_high_velocity_rate:.1%} high velocity vs {normal_high_velocity_rate:.1%} for normal cases',
                    'risk_factor': 'Medium',
                    'feature': 'velocity_score'
                })
        
        return {
            'rules': rules,
            'n_rules': len(rules),
            'high_risk_rules': len([r for r in rules if r['risk_factor'] == 'High']),
            'medium_risk_rules': len([r for r in rules if r['risk_factor'] == 'Medium']),
            'low_risk_rules': len([r for r in rules if r['risk_factor'] == 'Low'])
        }
    
    def plot_feature_importance(
        self,
        explanation: Dict[str, Any],
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance from explanation.
        
        Args:
            explanation: Explanation results.
            top_n: Number of top features to show.
            save_path: Path to save the plot.
        """
        if explanation.get('method') == 'shap' and 'shap_values' in explanation:
            values = explanation['shap_values']
        elif 'importance_scores' in explanation:
            values = explanation['importance_scores']
        else:
            print("No valid explanation data found")
            return
        
        # Get top features
        feature_importance = pd.DataFrame({
            'feature': explanation['feature_names'],
            'importance': values
        }).sort_values('importance', key=abs, ascending=False)
        
        top_features = feature_importance.head(top_n)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if explanation.get('method') == 'shap':
            colors = ['red' if x < 0 else 'blue' for x in top_features['importance']]
            plt.barh(range(len(top_features)), top_features['importance'], color=colors)
            plt.title(f'SHAP Feature Importance (Top {top_n})')
            plt.xlabel('SHAP Value')
        else:
            plt.barh(range(len(top_features)), top_features['importance'], color='blue')
            plt.title(f'Permutation Feature Importance (Top {top_n})')
            plt.xlabel('Importance Score')
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_waterfall(
        self,
        explanation: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """Plot SHAP waterfall chart.
        
        Args:
            explanation: SHAP explanation results.
            save_path: Path to save the plot.
        """
        if explanation.get('method') != 'shap' or 'shap_values' not in explanation:
            print("Waterfall plot only available for SHAP explanations")
            return
        
        if not SHAP_AVAILABLE:
            print("SHAP not available for waterfall plot")
            return
        
        try:
            # Create waterfall plot
            shap_values = explanation['shap_values']
            base_value = explanation['base_value']
            feature_names = explanation['feature_names']
            
            # Sort features by absolute SHAP value
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'shap_value': shap_values
            }).sort_values('shap_value', key=abs, ascending=False)
            
            # Create waterfall data
            cumulative_values = [base_value]
            for shap_val in feature_importance['shap_value']:
                cumulative_values.append(cumulative_values[-1] + shap_val)
            
            # Plot
            plt.figure(figsize=(12, 8))
            
            # Plot bars
            x_pos = range(len(cumulative_values))
            colors = ['gray'] + ['red' if x < 0 else 'blue' for x in feature_importance['shap_value']]
            
            plt.bar(x_pos, cumulative_values, color=colors, alpha=0.7)
            
            # Add labels
            plt.xticks(x_pos, ['Base'] + feature_importance['feature'].tolist(), rotation=45, ha='right')
            plt.ylabel('Cumulative SHAP Value')
            plt.title('SHAP Waterfall Plot')
            plt.grid(True, alpha=0.3)
            
            # Add final prediction
            final_prediction = cumulative_values[-1]
            plt.text(len(cumulative_values)-1, final_prediction, 
                    f'Final: {final_prediction:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating waterfall plot: {e}")


def create_explanation_report(
    explainer: IdentityTheftExplainer,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: str = "explanation_report.txt"
) -> str:
    """Create comprehensive explanation report.
    
    Args:
        explainer: Trained explainer.
        X: Feature matrix.
        y: True labels.
        output_path: Path to save the report.
        
    Returns:
        Report text.
    """
    # Generate rule evidence
    rule_evidence = explainer.generate_rule_evidence(X, y)
    
    # Generate batch explanation
    batch_explanation = explainer.explain_batch(X, n_samples=100)
    
    # Create report
    report = f"""
Identity Theft Detection Model Explanation Report
================================================

Model Type: {explainer.model.model_type if hasattr(explainer.model, 'model_type') else 'Unknown'}
Features Analyzed: {len(explainer.feature_names)}
Samples Explained: {batch_explanation['n_samples']}

Rule-Based Evidence:
-------------------
Number of Rules: {rule_evidence['n_rules']}
High Risk Rules: {rule_evidence['high_risk_rules']}
Medium Risk Rules: {rule_evidence['medium_risk_rules']}
Low Risk Rules: {rule_evidence['low_risk_rules']}

Rule Details:
"""
    
    for i, rule in enumerate(rule_evidence['rules'], 1):
        report += f"""
{i}. {rule['rule']} ({rule['risk_factor']} Risk)
   {rule['description']}
   Feature: {rule['feature']}
"""
    
    # Feature importance
    if batch_explanation['method'] == 'shap':
        report += f"""
Feature Importance (SHAP):
-------------------------
"""
        importance_data = pd.DataFrame({
            'feature': batch_explanation['feature_names'],
            'importance': batch_explanation['avg_shap_values']
        }).sort_values('importance', key=abs, ascending=False)
        
        for _, row in importance_data.head(10).iterrows():
            report += f"{row['feature']}: {row['importance']:.4f}\n"
    
    else:
        report += f"""
Feature Importance (Permutation):
--------------------------------
"""
        importance_data = pd.DataFrame({
            'feature': batch_explanation['feature_names'],
            'importance': batch_explanation['avg_importance_scores']
        }).sort_values('importance', ascending=False)
        
        for _, row in importance_data.head(10).iterrows():
            report += f"{row['feature']}: {row['importance']:.4f}\n"
    
    report += f"""
Recommendations:
---------------
"""
    
    # Add recommendations based on rules
    if rule_evidence['high_risk_rules'] > 0:
        report += "- High-risk patterns detected. Focus on suspicious login hours, device/location mismatches.\n"
    
    if rule_evidence['medium_risk_rules'] > 0:
        report += "- Medium-risk patterns detected. Monitor multiple login attempts and velocity scores.\n"
    
    if rule_evidence['n_rules'] < 3:
        report += "- Limited rule-based patterns found. Consider additional feature engineering.\n"
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    return report


if __name__ == "__main__":
    # Test the explainer
    from src.data.synthetic_data import generate_synthetic_data
    from src.models.identity_detector import IdentityTheftDetector
    
    # Generate data
    data = generate_synthetic_data(1000)
    
    # Train model
    detector = IdentityTheftDetector(model_type="xgboost")
    detector.fit(data['X_train'], data['y_train'])
    
    # Create explainer
    explainer = IdentityTheftExplainer(detector, data['feature_columns'])
    
    # Explain single prediction
    explanation = explainer.explain_prediction(data['X_test'], 0)
    print("Single Prediction Explanation:")
    print(f"Prediction: {explanation['prediction']:.4f}")
    print(f"Method: {explanation['method']}")
    
    # Generate rule evidence
    rule_evidence = explainer.generate_rule_evidence(data['X_test'], data['y_test'])
    print(f"\nRule Evidence:")
    print(f"Number of rules: {rule_evidence['n_rules']}")
    print(f"High risk rules: {rule_evidence['high_risk_rules']}")
    
    # Create report
    report = create_explanation_report(explainer, data['X_test'], data['y_test'])
    print(f"\nExplanation report created: explanation_report.txt")
