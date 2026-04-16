#!/usr/bin/env python3
"""Modernized Identity Theft Detection Demo.

This script demonstrates the modernized identity theft detection system
with advanced features, proper evaluation, and explainability.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from typing import Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from src.data.synthetic_data import generate_synthetic_data
from src.models.identity_detector import IdentityTheftDetector, evaluate_model
from src.eval.evaluator import IdentityTheftEvaluator
from src.explainability.explainer import IdentityTheftExplainer


def main():
    """Main demonstration function."""
    print("🛡️ Identity Theft Detection System - Modernized Demo")
    print("=" * 60)
    print("Defensive Research Demonstration - Educational Use Only")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic dataset
    print("\n📊 Generating synthetic dataset...")
    data = generate_synthetic_data(n_samples=5000)
    
    print(f"✅ Generated {data['metadata']['n_samples']:,} samples")
    print(f"   Fraud rate: {data['metadata']['fraud_rate']:.1%}")
    print(f"   Features: {data['metadata']['n_features']}")
    print(f"   Train samples: {len(data['X_train']):,}")
    print(f"   Test samples: {len(data['X_test']):,}")
    
    # Train multiple models for comparison
    models = {}
    results = {}
    
    model_types = ["random_forest", "xgboost", "lightgbm"]
    
    for model_type in model_types:
        print(f"\n🤖 Training {model_type.upper()} model...")
        
        # Initialize and train model
        detector = IdentityTheftDetector(model_type=model_type, random_state=42)
        detector.fit(
            data['X_train'], 
            data['y_train'],
            data['X_val'],
            data['y_val']
        )
        
        # Make predictions
        y_pred = detector.predict(data['X_test'])
        y_proba = detector.predict_proba(data['X_test'])[:, 1]
        
        # Evaluate model
        metrics = evaluate_model(detector, data['X_test'], data['y_test'])
        
        models[model_type] = detector
        results[model_type] = metrics
        
        print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"   AUC-PR: {metrics['auc_pr']:.4f}")
        print(f"   Precision@5%: {metrics['precision_at_5']:.4f}")
        print(f"   Alert Efficiency: {metrics['alert_efficiency']:.1%}")
    
    # Find best model
    best_model_type = max(results.keys(), key=lambda x: results[x]['auc_pr'])
    best_model = models[best_model_type]
    best_metrics = results[best_model_type]
    
    print(f"\n🏆 Best Model: {best_model_type.upper()}")
    print(f"   AUC-PR: {best_metrics['auc_pr']:.4f}")
    print(f"   Precision@5%: {best_metrics['precision_at_5']:.4f}")
    
    # Comprehensive evaluation
    print(f"\n📈 Comprehensive Evaluation of {best_model_type.upper()}")
    print("-" * 50)
    
    evaluator = IdentityTheftEvaluator(best_model_type.upper())
    y_pred = best_model.predict(data['X_test'])
    y_proba = best_model.predict_proba(data['X_test'])[:, 1]
    
    comprehensive_metrics = evaluator.evaluate(data['y_test'], y_pred, y_proba)
    
    # Display key metrics
    print("Performance Metrics:")
    print(f"  Accuracy: {comprehensive_metrics['accuracy']:.4f}")
    print(f"  Precision: {comprehensive_metrics['precision']:.4f}")
    print(f"  Recall: {comprehensive_metrics['recall']:.4f}")
    print(f"  F1-Score: {comprehensive_metrics['f1']:.4f}")
    print(f"  AUC-ROC: {comprehensive_metrics['auc_roc']:.4f}")
    print(f"  AUC-PR: {comprehensive_metrics['auc_pr']:.4f}")
    
    print("\nSecurity Metrics:")
    print(f"  Precision@1%: {comprehensive_metrics['precision_at_1']:.4f}")
    print(f"  Precision@5%: {comprehensive_metrics['precision_at_5']:.4f}")
    print(f"  Precision@10%: {comprehensive_metrics['precision_at_10']:.4f}")
    print(f"  Precision@20%: {comprehensive_metrics['precision_at_20']:.4f}")
    
    print("\nAlert Workload Metrics:")
    print(f"  Alert Rate: {comprehensive_metrics['alert_rate']:.2%}")
    print(f"  Alerts per 1000: {comprehensive_metrics['alerts_per_1000']:.1f}")
    print(f"  Alert Efficiency: {comprehensive_metrics['alert_efficiency']:.1%}")
    print(f"  False Alerts per 1000: {comprehensive_metrics['false_alerts_per_1000']:.1f}")
    
    print("\nCalibration Metrics:")
    print(f"  Expected Calibration Error: {comprehensive_metrics['expected_calibration_error']:.4f}")
    print(f"  Maximum Calibration Error: {comprehensive_metrics['maximum_calibration_error']:.4f}")
    
    # Feature importance analysis
    print(f"\n🔍 Feature Importance Analysis")
    print("-" * 50)
    
    importance = best_model.get_feature_importance()
    if importance.sum() > 0:
        top_features = importance.head(10)
        print("Top 10 Most Important Features:")
        for i, (feature, score) in enumerate(top_features.items(), 1):
            print(f"  {i:2d}. {feature:<25} {score:.4f}")
    else:
        print("Feature importance not available for this model type.")
    
    # Explainability analysis
    print(f"\n🧠 Explainability Analysis")
    print("-" * 50)
    
    try:
        explainer = IdentityTheftExplainer(best_model, data['feature_columns'])
        
        # Generate rule evidence
        rule_evidence = explainer.generate_rule_evidence(data['X_test'], data['y_test'])
        
        print(f"Rule-Based Evidence:")
        print(f"  Total Rules: {rule_evidence['n_rules']}")
        print(f"  High Risk Rules: {rule_evidence['high_risk_rules']}")
        print(f"  Medium Risk Rules: {rule_evidence['medium_risk_rules']}")
        print(f"  Low Risk Rules: {rule_evidence['low_risk_rules']}")
        
        if rule_evidence['rules']:
            print("\nDetected Risk Patterns:")
            for i, rule in enumerate(rule_evidence['rules'][:5], 1):
                print(f"  {i}. {rule['rule']} ({rule['risk_factor']} Risk)")
                print(f"     {rule['description']}")
        
        # Explain a high-risk case
        high_risk_indices = np.where(y_proba > 0.8)[0]
        if len(high_risk_indices) > 0:
            print(f"\nHigh-Risk Case Explanation:")
            explanation = explainer.explain_prediction(data['X_test'], high_risk_indices[0])
            print(f"  Risk Score: {explanation['prediction']:.4f}")
            print(f"  Explanation Method: {explanation['method']}")
            
            if explanation['method'] == 'shap' and 'shap_values' in explanation:
                # Show top contributing features
                shap_df = pd.DataFrame({
                    'feature': explanation['feature_names'],
                    'shap_value': explanation['shap_values']
                }).sort_values('shap_value', key=abs, ascending=False)
                
                print("  Top Contributing Features:")
                for _, row in shap_df.head(5).iterrows():
                    direction = "increases" if row['shap_value'] > 0 else "decreases"
                    print(f"    {row['feature']}: {direction} risk by {abs(row['shap_value']):.4f}")
    
    except Exception as e:
        print(f"Explainability analysis failed: {e}")
    
    # Model comparison
    print(f"\n📊 Model Comparison")
    print("-" * 50)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df[['auc_roc', 'auc_pr', 'precision_at_5', 'alert_efficiency']].round(4)
    
    print("Model Performance Comparison:")
    print(comparison_df.to_string())
    
    # Generate detailed report
    print(f"\n📋 Generating Detailed Report...")
    
    report = evaluator.generate_report(data['y_test'], y_pred, y_proba)
    
    # Save report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/identity_theft_evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Detailed report saved to: {report_path}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/best_{best_model_type}_model.pkl"
    best_model.save_model(model_path)
    print(f"✅ Best model saved to: {model_path}")
    
    # Final summary
    print(f"\n🎯 Final Summary")
    print("=" * 60)
    print(f"Best Model: {best_model_type.upper()}")
    print(f"AUC-PR: {best_metrics['auc_pr']:.4f}")
    print(f"Precision@5%: {best_metrics['precision_at_5']:.4f}")
    print(f"Alert Efficiency: {best_metrics['alert_efficiency']:.1%}")
    print(f"Expected Calibration Error: {comprehensive_metrics['expected_calibration_error']:.4f}")
    
    if comprehensive_metrics['alert_efficiency'] > 0.7:
        print("✅ Model shows good alert efficiency")
    else:
        print("⚠️ Model may need improvement in alert efficiency")
    
    if comprehensive_metrics['expected_calibration_error'] < 0.1:
        print("✅ Model is well-calibrated")
    else:
        print("⚠️ Model may need calibration improvement")
    
    print("\n🚀 To run the interactive demo:")
    print("   streamlit run demo/app.py")
    
    print("\n⚠️ DISCLAIMER: This is a defensive research demonstration for educational purposes only.")
    print("   Results may be inaccurate and should not be used for production security operations.")


if __name__ == "__main__":
    main()
