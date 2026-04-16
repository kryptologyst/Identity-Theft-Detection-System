#!/usr/bin/env python3
"""Project 886: Identity Theft Detection - Modernized Implementation

This is a modernized version of the original identity theft detection system
with enhanced features, proper evaluation, and explainability.

DISCLAIMER: This is a defensive research demonstration for educational purposes only.
Results may be inaccurate and should not be used for production security operations.
"""

import sys
import os
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modernized modules
try:
    from src.data.synthetic_data import generate_synthetic_data
    from src.models.identity_detector import IdentityTheftDetector, evaluate_model
    from src.eval.evaluator import IdentityTheftEvaluator
    MODERN_MODULES_AVAILABLE = True
except ImportError:
    MODERN_MODULES_AVAILABLE = False
    print("Warning: Modern modules not available. Using basic implementation.")


def original_implementation():
    """Original basic implementation for comparison."""
    print("🔄 Running Original Implementation")
    print("-" * 40)
    
    # Simulated identity-related transaction logs
    data = {
        'LoginHour': [10, 2, 14, 1, 13, 0, 15, 3],             # time of login
        'DeviceMatch': [1, 0, 1, 0, 1, 0, 1, 0],               # known device (1) or not (0)
        'LocationMatch': [1, 0, 1, 0, 1, 0, 1, 0],             # known location (1) or not (0)
        'MultipleAttempts': [0, 1, 0, 1, 0, 1, 0, 1],          # repeated login attempts
        'IdentityTheft': [0, 1, 0, 1, 0, 1, 0, 1]              # 1 = identity theft, 0 = normal
    }
    
    df = pd.DataFrame(data)
    
    # Features and target
    X = df.drop('IdentityTheft', axis=1)
    y = df['IdentityTheft']
    
    # Split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train the identity theft detection model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Identity Theft Detection Report:")
    print(classification_report(y_test, y_pred))
    
    # Predict on new login behavior
    new_login = pd.DataFrame([{
        'LoginHour': 1,
        'DeviceMatch': 0,
        'LocationMatch': 0,
        'MultipleAttempts': 1
    }])
    
    risk_score = model.predict_proba(new_login)[0][1]
    print(f"\nPredicted Identity Theft Risk: {risk_score:.2%}")
    
    return model, X_test, y_test, y_pred


def modernized_implementation():
    """Modernized implementation with advanced features."""
    print("🚀 Running Modernized Implementation")
    print("-" * 40)
    
    # Generate synthetic dataset
    print("📊 Generating synthetic dataset...")
    data = generate_synthetic_data(n_samples=5000)
    
    print(f"✅ Generated {data['metadata']['n_samples']:,} samples")
    print(f"   Fraud rate: {data['metadata']['fraud_rate']:.1%}")
    print(f"   Features: {data['metadata']['n_features']}")
    
    # Train modernized model
    print("\n🤖 Training modernized model...")
    detector = IdentityTheftDetector(model_type="xgboost", random_state=42)
    detector.fit(
        data['X_train'], 
        data['y_train'],
        data['X_val'],
        data['y_val']
    )
    
    # Make predictions
    y_pred = detector.predict(data['X_test'])
    y_proba = detector.predict_proba(data['X_test'])[:, 1]
    
    # Comprehensive evaluation
    print("\n📈 Comprehensive Evaluation:")
    evaluator = IdentityTheftEvaluator("XGBoost")
    metrics = evaluator.evaluate(data['y_test'], y_pred, y_proba)
    
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"  Precision@5%: {metrics['precision_at_5']:.4f}")
    print(f"  Alert Efficiency: {metrics['alert_efficiency']:.1%}")
    
    # Feature importance
    print("\n🔍 Top 5 Most Important Features:")
    importance = detector.get_feature_importance()
    if importance.sum() > 0:
        top_features = importance.head(5)
        for i, (feature, score) in enumerate(top_features.items(), 1):
            print(f"  {i}. {feature:<25} {score:.4f}")
    
    # Rule-based evidence
    print("\n🧠 Rule-Based Evidence:")
    from src.explainability.explainer import IdentityTheftExplainer
    explainer = IdentityTheftExplainer(detector, data['feature_columns'])
    rule_evidence = explainer.generate_rule_evidence(data['X_test'], data['y_test'])
    
    print(f"  Total Rules: {rule_evidence['n_rules']}")
    print(f"  High Risk Rules: {rule_evidence['high_risk_rules']}")
    print(f"  Medium Risk Rules: {rule_evidence['medium_risk_rules']}")
    
    if rule_evidence['rules']:
        print("\nDetected Risk Patterns:")
        for i, rule in enumerate(rule_evidence['rules'][:3], 1):
            print(f"  {i}. {rule['rule']} ({rule['risk_factor']} Risk)")
    
    return detector, data['X_test'], data['y_test'], y_pred, metrics


def compare_implementations():
    """Compare original and modernized implementations."""
    print("\n📊 Implementation Comparison")
    print("=" * 50)
    
    # Run original implementation
    original_model, orig_X_test, orig_y_test, orig_y_pred = original_implementation()
    
    print("\n" + "="*50)
    
    # Run modernized implementation
    if MODERN_MODULES_AVAILABLE:
        modern_model, mod_X_test, mod_y_test, mod_y_pred, metrics = modernized_implementation()
        
        print("\n📈 Comparison Summary:")
        print("-" * 30)
        print("Original Implementation:")
        print(f"  Dataset Size: {len(orig_X_test)} samples")
        print(f"  Features: {orig_X_test.shape[1]}")
        print(f"  Model: Random Forest")
        
        print("\nModernized Implementation:")
        print(f"  Dataset Size: {len(mod_X_test):,} samples")
        print(f"  Features: {mod_X_test.shape[1]}")
        print(f"  Model: XGBoost")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
        print(f"  Precision@5%: {metrics['precision_at_5']:.4f}")
        print(f"  Alert Efficiency: {metrics['alert_efficiency']:.1%}")
        
        print("\n✅ Modernized implementation provides:")
        print("  - Larger, more realistic synthetic dataset")
        print("  - Advanced machine learning models")
        print("  - Comprehensive security-relevant metrics")
        print("  - Feature importance analysis")
        print("  - Rule-based explainability")
        print("  - Proper train/validation/test splits")
        print("  - Time-aware data splitting")
        print("  - Graph-based features")
        print("  - Calibration analysis")
    else:
        print("⚠️ Modern modules not available. Install dependencies to see full comparison.")


def main():
    """Main function."""
    print("🛡️ Identity Theft Detection System")
    print("=" * 50)
    print("Defensive Research Demonstration - Educational Use Only")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Compare implementations
    compare_implementations()
    
    print("\n🚀 Next Steps:")
    print("  1. Run the interactive demo: streamlit run demo/app.py")
    print("  2. Train custom models: python scripts/train.py")
    print("  3. Evaluate models: python scripts/evaluate.py")
    print("  4. Run tests: python -m pytest tests/")
    
    print("\n⚠️ DISCLAIMER:")
    print("This is a defensive research demonstration for educational purposes only.")
    print("Results may be inaccurate and should not be used for production security operations.")
    print("This is NOT a SOC (Security Operations Center) tool.")


if __name__ == "__main__":
    main()

