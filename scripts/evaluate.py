#!/usr/bin/env python3
"""Evaluation script for Identity Theft Detection models.

This script evaluates trained models and generates comprehensive reports.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.synthetic_data import generate_synthetic_data
from src.models.identity_detector import IdentityTheftDetector
from src.eval.evaluator import IdentityTheftEvaluator


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/evaluation.log'),
            logging.StreamHandler()
        ]
    )


def evaluate_model_comprehensive(
    model_path: str,
    dataset_size: int = 10000,
    output_dir: str = "evaluation_results"
) -> None:
    """Perform comprehensive model evaluation.
    
    Args:
        model_path: Path to trained model.
        dataset_size: Size of test dataset.
        output_dir: Output directory for results.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Loading model from {model_path}")
    
    # Load model
    detector = IdentityTheftDetector.load_model(model_path)
    
    logging.info(f"Model type: {detector.model_type}")
    
    # Generate test data
    logging.info(f"Generating {dataset_size:,} test samples...")
    data = generate_synthetic_data(dataset_size)
    
    # Make predictions
    logging.info("Making predictions...")
    y_pred = detector.predict(data['X_test'])
    y_proba = detector.predict_proba(data['X_test'])[:, 1]
    
    # Evaluate model
    logging.info("Evaluating model performance...")
    evaluator = IdentityTheftEvaluator(detector.model_type.upper())
    metrics = evaluator.evaluate(data['y_test'], y_pred, y_proba)
    
    # Generate visualizations
    logging.info("Generating visualizations...")
    
    # ROC Curve
    roc_fig = evaluator.plot_roc_curve(data['y_test'], y_proba)
    roc_fig.write_html(os.path.join(output_dir, "roc_curve.html"))
    roc_fig.write_image(os.path.join(output_dir, "roc_curve.png"))
    
    # Precision-Recall Curve
    pr_fig = evaluator.plot_precision_recall_curve(data['y_test'], y_proba)
    pr_fig.write_html(os.path.join(output_dir, "pr_curve.html"))
    pr_fig.write_image(os.path.join(output_dir, "pr_curve.png"))
    
    # Calibration Curve
    cal_fig = evaluator.plot_calibration_curve(data['y_test'], y_proba)
    cal_fig.write_html(os.path.join(output_dir, "calibration_curve.html"))
    cal_fig.write_image(os.path.join(output_dir, "calibration_curve.png"))
    
    # Confusion Matrix
    cm_fig = evaluator.plot_confusion_matrix(data['y_test'], y_pred)
    cm_fig.write_html(os.path.join(output_dir, "confusion_matrix.html"))
    cm_fig.write_image(os.path.join(output_dir, "confusion_matrix.png"))
    
    # Feature Importance
    importance = detector.get_feature_importance()
    if importance.sum() > 0:
        plt.figure(figsize=(10, 8))
        top_features = importance.head(15)
        plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Importance Score')
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Risk Score Distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(y_proba[data['y_test'] == 0], bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(y_proba[data['y_test'] == 1], bins=50, alpha=0.7, label='Fraud', color='red')
    plt.xlabel('Risk Score')
    plt.ylabel('Count')
    plt.title('Risk Score Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        if y_pred_thresh.sum() > 0:
            precision = (y_pred_thresh & data['y_test']).sum() / y_pred_thresh.sum()
            recall = (y_pred_thresh & data['y_test']).sum() / data['y_test'].sum()
        else:
            precision = 0
            recall = 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    plt.plot(thresholds, precisions, label='Precision', color='blue')
    plt.plot(thresholds, recalls, label='Recall', color='red')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate comprehensive report
    logging.info("Generating comprehensive report...")
    report = evaluator.generate_report(data['y_test'], y_pred, y_proba)
    
    # Save report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save metrics as YAML
    metrics_path = os.path.join(output_dir, "metrics.yaml")
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'y_true': data['y_test'],
        'y_pred': y_pred,
        'y_proba': y_proba
    })
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    logging.info(f"Evaluation completed. Results saved to {output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model Type: {detector.model_type}")
    print(f"Test Dataset Size: {len(data['y_test']):,}")
    print(f"Fraud Rate: {data['y_test'].mean():.1%}")
    print()
    print("Performance Metrics:")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print()
    print("Security Metrics:")
    print(f"  Precision@1%: {metrics['precision_at_1']:.4f}")
    print(f"  Precision@5%: {metrics['precision_at_5']:.4f}")
    print(f"  Precision@10%: {metrics['precision_at_10']:.4f}")
    print()
    print("Alert Metrics:")
    print(f"  Alert Rate: {metrics['alert_rate']:.2%}")
    print(f"  Alert Efficiency: {metrics['alert_efficiency']:.1%}")
    print(f"  Alerts per 1000: {metrics['alerts_per_1000']:.1f}")
    print()
    print(f"Results saved to: {output_dir}")
    print("="*60)


def compare_models(
    model_paths: list,
    dataset_size: int = 10000,
    output_dir: str = "model_comparison"
) -> None:
    """Compare multiple models.
    
    Args:
        model_paths: List of paths to trained models.
        dataset_size: Size of test dataset.
        output_dir: Output directory for results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Comparing {len(model_paths)} models...")
    
    # Generate test data
    data = generate_synthetic_data(dataset_size)
    
    results = {}
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('.pkl', '')
        logging.info(f"Evaluating {model_name}...")
        
        # Load model
        detector = IdentityTheftDetector.load_model(model_path)
        
        # Make predictions
        y_pred = detector.predict(data['X_test'])
        y_proba = detector.predict_proba(data['X_test'])[:, 1]
        
        # Evaluate
        evaluator = IdentityTheftEvaluator(model_name)
        metrics = evaluator.evaluate(data['y_test'], y_pred, y_proba)
        
        results[model_name] = metrics
    
    # Create comparison table
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)
    
    # Save comparison
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"))
    
    # Create comparison plots
    metrics_to_plot = ['auc_roc', 'auc_pr', 'precision_at_5', 'alert_efficiency']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in comparison_df.columns:
            comparison_df[metric].plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Model comparison completed. Results saved to {output_dir}")
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(comparison_df)
    print("="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Identity Theft Detection Model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=10000,
        help="Size of test dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--compare-models",
        nargs='+',
        help="Paths to multiple models for comparison"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if args.compare_models:
        # Compare multiple models
        compare_models(args.compare_models, args.dataset_size, args.output_dir)
    else:
        # Evaluate single model
        evaluate_model_comprehensive(args.model_path, args.dataset_size, args.output_dir)


if __name__ == "__main__":
    main()
