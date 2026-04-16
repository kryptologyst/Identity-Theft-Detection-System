# Identity Theft Detection System

## DISCLAIMER

**This is a defensive research demonstration project for educational purposes only.**

- This system is designed for research and educational use
- Results may be inaccurate and should not be used for production security operations
- This is NOT a SOC (Security Operations Center) tool
- No offensive capabilities or exploitation techniques are implemented
- All data is synthetic and de-identified for demonstration purposes

## Overview

This project implements an Identity Theft Detection system that analyzes user behavior patterns to identify suspicious account access attempts. The system focuses on detecting account takeover patterns through behavioral analysis, device fingerprinting, and location-based anomaly detection.

## Features

- **Behavioral Analysis**: Login patterns, device usage, location tracking
- **Graph-based Features**: Entity relationships, device/IP associations
- **Anomaly Detection**: Unusual access patterns and timing
- **Risk Scoring**: Calibrated probability scores for identity theft attempts
- **Explainable AI**: SHAP-based feature importance and rule-based evidence
- **Interactive Demo**: Streamlit-based web interface for testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Identity-Theft-Detection-System.git
cd Identity-Theft-Detection-System

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models.identity_detector import IdentityTheftDetector
from src.data.synthetic_data import generate_synthetic_data

# Generate synthetic data
data = generate_synthetic_data(n_samples=10000)

# Initialize detector
detector = IdentityTheftDetector()

# Train model
detector.fit(data['X_train'], data['y_train'])

# Predict on new data
risk_scores = detector.predict_proba(data['X_test'])
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Dataset Schema

The system expects transaction logs with the following features:

- **Temporal Features**: Login hour, day of week, time since last login
- **Device Features**: Device fingerprint, browser type, OS version
- **Location Features**: IP geolocation, GPS coordinates, location history
- **Behavioral Features**: Transaction patterns, session duration, click patterns
- **Graph Features**: Device associations, IP relationships, account linkages

All personally identifiable information (PII) is hashed or obfuscated in the synthetic datasets.

## Training and Evaluation

### Training

```bash
# Train with default configuration
python scripts/train.py --config configs/default.yaml

# Train with custom parameters
python scripts/train.py --config configs/custom.yaml --epochs 100
```

### Evaluation

```bash
# Run evaluation suite
python scripts/evaluate.py --model-path models/best_model.pkl

# Generate evaluation report
python scripts/generate_report.py --output-dir reports/
```

## Metrics

The system evaluates performance using security-relevant metrics:

- **AUCPR**: Area Under Precision-Recall Curve (primary metric for rare events)
- **Precision@K**: Precision at top K% of predictions
- **Alert Workload**: Number of alerts per 1000 events
- **False Positive Rate**: At target True Positive Rate
- **Detection Latency**: Time to detection for streaming scenarios

## Model Architecture

### Baseline Models
- Random Forest with class balancing
- XGBoost with custom loss functions
- LightGBM with categorical features

### Advanced Models
- Graph Neural Networks for entity relationships
- Sequence models for temporal patterns
- Ensemble methods combining multiple approaches

## Configuration

The system uses YAML-based configuration files:

```yaml
# configs/default.yaml
data:
  train_size: 0.7
  val_size: 0.15
  test_size: 0.15
  
model:
  type: "xgboost"
  params:
    n_estimators: 100
    max_depth: 6
    
evaluation:
  metrics: ["aucpr", "precision_at_k", "alert_workload"]
  k_values: [1, 5, 10]
```

## Safety and Privacy

- **PII Protection**: All sensitive data is hashed or obfuscated
- **Data Retention**: Synthetic data only, no real user data stored
- **Audit Logging**: All model predictions and decisions are logged
- **Access Control**: Demo system with no production access

## Limitations

- Synthetic data may not reflect real-world patterns
- Model performance on synthetic data may not generalize
- No real-time threat intelligence integration
- Limited to demonstrated attack patterns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about this research demonstration, please contact the development team.
# Identity-Theft-Detection-System
