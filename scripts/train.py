#!/usr/bin/env python3
"""Training script for Identity Theft Detection models.

This script trains various machine learning models for identity theft detection
using synthetic data and saves the best performing model.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
import joblib
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.synthetic_data import generate_synthetic_data
from src.models.identity_detector import IdentityTheftDetector, evaluate_model
from src.eval.evaluator import IdentityTheftEvaluator


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration.
    
    Args:
        config: Logging configuration dictionary.
    """
    log_config = config.get('logging', {})
    
    # Create logs directory
    log_file = log_config.get('file', 'logs/training.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_model(
    config: Dict[str, Any],
    model_type: str = None
) -> IdentityTheftDetector:
    """Train identity theft detection model.
    
    Args:
        config: Configuration dictionary.
        model_type: Override model type from config.
        
    Returns:
        Trained model.
    """
    # Get model configuration
    model_config = config['model']
    data_config = config['data']
    
    # Override model type if provided
    if model_type:
        model_config['type'] = model_type
    
    logging.info(f"Training {model_config['type']} model...")
    
    # Generate synthetic data
    logging.info(f"Generating {data_config['n_samples']:,} synthetic samples...")
    data = generate_synthetic_data(data_config['n_samples'])
    
    logging.info(f"Dataset generated with {data['metadata']['fraud_rate']:.1%} fraud rate")
    
    # Initialize model
    detector = IdentityTheftDetector(
        model_type=model_config['type'],
        random_state=data_config['random_seed']
    )
    
    # Train model
    logging.info("Training model...")
    detector.fit(
        data['X_train'], 
        data['y_train'],
        data['X_val'],
        data['y_val']
    )
    
    logging.info("Model training completed")
    
    # Evaluate model
    logging.info("Evaluating model...")
    metrics = evaluate_model(detector, data['X_test'], data['y_test'])
    
    # Log key metrics
    logging.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    logging.info(f"AUC-PR: {metrics['auc_pr']:.4f}")
    logging.info(f"Precision@5%: {metrics['precision_at_5']:.4f}")
    logging.info(f"Alert Efficiency: {metrics['alert_efficiency']:.1%}")
    
    return detector, metrics, data


def save_model_and_results(
    detector: IdentityTheftDetector,
    metrics: Dict[str, float],
    data: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str = "models"
) -> None:
    """Save trained model and evaluation results.
    
    Args:
        detector: Trained model.
        metrics: Evaluation metrics.
        data: Dataset used for training.
        config: Configuration dictionary.
        output_dir: Output directory.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "best_model.pkl")
    detector.save_model(model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.yaml")
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    logging.info(f"Metrics saved to {metrics_path}")
    
    # Save feature importance
    importance = detector.get_feature_importance()
    if importance.sum() > 0:
        importance_path = os.path.join(output_dir, "feature_importance.csv")
        importance.to_csv(importance_path)
        logging.info(f"Feature importance saved to {importance_path}")
    
    # Generate detailed report
    evaluator = IdentityTheftEvaluator(config['model']['type'].upper())
    y_pred = detector.predict(data['X_test'])
    y_proba = detector.predict_proba(data['X_test'])[:, 1]
    
    report = evaluator.generate_report(data['y_test'], y_pred, y_proba)
    
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    logging.info(f"Evaluation report saved to {report_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Identity Theft Detection Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["random_forest", "xgboost", "lightgbm", "neural_net"],
        help="Override model type from config"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for saved model and results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
    setup_logging(config)
    
    logging.info("Starting Identity Theft Detection model training")
    logging.info(f"Configuration: {args.config}")
    logging.info(f"Model type: {args.model_type or config['model']['type']}")
    logging.info(f"Output directory: {args.output_dir}")
    
    try:
        # Train model
        detector, metrics, data = train_model(config, args.model_type)
        
        # Save results
        save_model_and_results(detector, metrics, data, config, args.output_dir)
        
        logging.info("Training completed successfully")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Model Type: {config['model']['type']}")
        print(f"Dataset Size: {data['metadata']['n_samples']:,}")
        print(f"Fraud Rate: {data['metadata']['fraud_rate']:.1%}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"AUC-PR: {metrics['auc_pr']:.4f}")
        print(f"Precision@5%: {metrics['precision_at_5']:.4f}")
        print(f"Alert Efficiency: {metrics['alert_efficiency']:.1%}")
        print(f"Model saved to: {args.output_dir}/best_model.pkl")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
