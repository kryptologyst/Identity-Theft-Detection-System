"""Test suite for Identity Theft Detection system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.synthetic_data import IdentityTheftDataGenerator, generate_synthetic_data
from src.models.identity_detector import IdentityTheftDetector, IdentityTheftNeuralNet
from src.eval.evaluator import IdentityTheftEvaluator
from src.explainability.explainer import IdentityTheftExplainer


class TestDataGeneration:
    """Test data generation functionality."""
    
    def test_data_generator_initialization(self):
        """Test data generator initialization."""
        generator = IdentityTheftDataGenerator(seed=42)
        assert generator.seed == 42
        assert len(generator.device_fingerprints) == 1000
        assert len(generator.locations) == 500
        assert 'normal_login_hours' in generator.user_patterns
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        data = generate_synthetic_data(n_samples=1000)
        
        # Check data structure
        assert 'X_train' in data
        assert 'X_val' in data
        assert 'X_test' in data
        assert 'y_train' in data
        assert 'y_val' in data
        assert 'y_test' in data
        assert 'feature_columns' in data
        assert 'metadata' in data
        
        # Check data sizes
        assert len(data['X_train']) > 0
        assert len(data['X_val']) > 0
        assert len(data['X_test']) > 0
        
        # Check metadata
        assert data['metadata']['n_samples'] == 1000
        assert data['metadata']['fraud_rate'] > 0
        assert data['metadata']['fraud_rate'] < 1
    
    def test_data_consistency(self):
        """Test data consistency across runs."""
        data1 = generate_synthetic_data(n_samples=100, seed=42)
        data2 = generate_synthetic_data(n_samples=100, seed=42)
        
        # Should be identical with same seed
        pd.testing.assert_frame_equal(data1['X_train'], data2['X_train'])
        pd.testing.assert_series_equal(data1['y_train'], data2['y_train'])


class TestModels:
    """Test model functionality."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        detector = IdentityTheftDetector(model_type="xgboost", random_state=42)
        assert detector.model_type == "xgboost"
        assert detector.random_state == 42
        assert not detector.is_fitted
    
    def test_model_training(self):
        """Test model training."""
        data = generate_synthetic_data(n_samples=1000)
        
        detector = IdentityTheftDetector(model_type="random_forest", random_state=42)
        detector.fit(data['X_train'], data['y_train'])
        
        assert detector.is_fitted
        assert detector.feature_columns is not None
        assert len(detector.feature_columns) > 0
    
    def test_model_prediction(self):
        """Test model prediction."""
        data = generate_synthetic_data(n_samples=1000)
        
        detector = IdentityTheftDetector(model_type="random_forest", random_state=42)
        detector.fit(data['X_train'], data['y_train'])
        
        # Test prediction
        y_pred = detector.predict(data['X_test'])
        y_proba = detector.predict_proba(data['X_test'])
        
        assert len(y_pred) == len(data['X_test'])
        assert len(y_proba) == len(data['X_test'])
        assert y_proba.shape[1] == 2  # Binary classification
        assert np.all(y_proba >= 0) and np.all(y_proba <= 1)
        assert np.allclose(y_proba.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_feature_importance(self):
        """Test feature importance."""
        data = generate_synthetic_data(n_samples=1000)
        
        detector = IdentityTheftDetector(model_type="random_forest", random_state=42)
        detector.fit(data['X_train'], data['y_train'])
        
        importance = detector.get_feature_importance()
        assert len(importance) == len(data['feature_columns'])
        assert np.all(importance >= 0)
        assert np.allclose(importance.sum(), 1.0)  # Importance sums to 1


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = IdentityTheftEvaluator("TestModel")
        assert evaluator.model_name == "TestModel"
        assert evaluator.results == {}
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        # Generate test data
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.1, 1000)
        y_pred = np.random.binomial(1, 0.1, 1000)
        y_proba = np.random.uniform(0, 1, 1000)
        
        evaluator = IdentityTheftEvaluator("TestModel")
        metrics = evaluator.evaluate(y_true, y_pred, y_proba)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr',
            'precision_at_1', 'precision_at_5', 'precision_at_10', 'precision_at_20',
            'alert_rate', 'alerts_per_1000', 'alert_efficiency', 'false_alerts_per_1000',
            'expected_calibration_error', 'maximum_calibration_error'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_report_generation(self):
        """Test report generation."""
        # Generate test data
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.1, 1000)
        y_pred = np.random.binomial(1, 0.1, 1000)
        y_proba = np.random.uniform(0, 1, 1000)
        
        evaluator = IdentityTheftEvaluator("TestModel")
        report = evaluator.generate_report(y_true, y_pred, y_proba)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "TestModel" in report
        assert "Performance Metrics" in report
        assert "Security Metrics" in report


class TestExplainability:
    """Test explainability functionality."""
    
    def test_explainer_initialization(self):
        """Test explainer initialization."""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        
        feature_names = ['feature1', 'feature2', 'feature3']
        explainer = IdentityTheftExplainer(mock_model, feature_names)
        
        assert explainer.model == mock_model
        assert explainer.feature_names == feature_names
    
    def test_rule_evidence_generation(self):
        """Test rule evidence generation."""
        # Generate test data
        data = generate_synthetic_data(n_samples=1000)
        
        detector = IdentityTheftDetector(model_type="random_forest", random_state=42)
        detector.fit(data['X_train'], data['y_train'])
        
        explainer = IdentityTheftExplainer(detector, data['feature_columns'])
        rule_evidence = explainer.generate_rule_evidence(data['X_test'], data['y_test'])
        
        assert 'rules' in rule_evidence
        assert 'n_rules' in rule_evidence
        assert 'high_risk_rules' in rule_evidence
        assert 'medium_risk_rules' in rule_evidence
        assert 'low_risk_rules' in rule_evidence
        
        assert isinstance(rule_evidence['rules'], list)
        assert rule_evidence['n_rules'] >= 0
    
    def test_prediction_explanation(self):
        """Test prediction explanation."""
        # Generate test data
        data = generate_synthetic_data(n_samples=1000)
        
        detector = IdentityTheftDetector(model_type="random_forest", random_state=42)
        detector.fit(data['X_train'], data['y_train'])
        
        explainer = IdentityTheftExplainer(detector, data['feature_columns'])
        explanation = explainer.explain_prediction(data['X_test'], 0)
        
        assert 'prediction' in explanation
        assert 'method' in explanation
        assert explanation['prediction'] >= 0
        assert explanation['prediction'] <= 1


class TestIntegration:
    """Test integration between components."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        data = generate_synthetic_data(n_samples=1000)
        
        # Train model
        detector = IdentityTheftDetector(model_type="random_forest", random_state=42)
        detector.fit(data['X_train'], data['y_train'])
        
        # Make predictions
        y_pred = detector.predict(data['X_test'])
        y_proba = detector.predict_proba(data['X_test'])[:, 1]
        
        # Evaluate
        evaluator = IdentityTheftEvaluator("RandomForest")
        metrics = evaluator.evaluate(data['y_test'], y_pred, y_proba)
        
        # Explain
        explainer = IdentityTheftExplainer(detector, data['feature_columns'])
        rule_evidence = explainer.generate_rule_evidence(data['X_test'], data['y_test'])
        
        # Check that everything worked
        assert metrics['auc_roc'] > 0
        assert metrics['auc_pr'] > 0
        assert rule_evidence['n_rules'] >= 0
    
    def test_model_saving_loading(self):
        """Test model saving and loading."""
        # Generate data and train model
        data = generate_synthetic_data(n_samples=1000)
        detector = IdentityTheftDetector(model_type="random_forest", random_state=42)
        detector.fit(data['X_train'], data['y_train'])
        
        # Save model
        model_path = "test_model.pkl"
        detector.save_model(model_path)
        
        # Load model
        loaded_detector = IdentityTheftDetector.load_model(model_path)
        
        # Check that loaded model works
        y_pred_original = detector.predict(data['X_test'])
        y_pred_loaded = loaded_detector.predict(data['X_test'])
        
        np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
        
        # Clean up
        os.remove(model_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
