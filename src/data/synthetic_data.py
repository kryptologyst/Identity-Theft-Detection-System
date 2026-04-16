"""Synthetic data generation for identity theft detection.

This module generates realistic synthetic datasets for identity theft detection
research and demonstration purposes. All data is completely synthetic and
contains no real user information.
"""

import hashlib
import random
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from faker import Faker
from sklearn.model_selection import train_test_split


class IdentityTheftDataGenerator:
    """Generates synthetic identity theft detection datasets.
    
    This class creates realistic synthetic data for identity theft detection
    including user behavior patterns, device fingerprints, location data,
    and transaction logs. All data is completely synthetic.
    """
    
    def __init__(self, seed: int = 42) -> None:
        """Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Device fingerprints (synthetic)
        self.device_fingerprints = self._generate_device_fingerprints()
        
        # Location data (synthetic)
        self.locations = self._generate_locations()
        
        # User behavior patterns
        self.user_patterns = self._generate_user_patterns()
    
    def _generate_device_fingerprints(self) -> List[str]:
        """Generate synthetic device fingerprints."""
        devices = []
        for _ in range(1000):
            # Create synthetic device fingerprint
            device_id = f"device_{random.randint(100000, 999999)}"
            devices.append(device_id)
        return devices
    
    def _generate_locations(self) -> List[Dict[str, Any]]:
        """Generate synthetic location data."""
        locations = []
        for _ in range(500):
            location = {
                'city': self.fake.city(),
                'country': self.fake.country(),
                'latitude': float(self.fake.latitude()),
                'longitude': float(self.fake.longitude()),
                'ip_range': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.0/24"
            }
            locations.append(location)
        return locations
    
    def _generate_user_patterns(self) -> Dict[str, Any]:
        """Generate synthetic user behavior patterns."""
        return {
            'normal_login_hours': list(range(6, 23)),  # 6 AM to 11 PM
            'suspicious_login_hours': list(range(0, 6)) + list(range(23, 24)),  # Midnight to 6 AM
            'normal_session_duration': (30, 180),  # 30-180 minutes
            'suspicious_session_duration': (5, 30),  # 5-30 minutes
            'normal_transaction_amount': (10, 1000),  # $10-$1000
            'suspicious_transaction_amount': (1000, 10000),  # $1000-$10000
        }
    
    def _hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for privacy protection."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def generate_transaction_logs(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic transaction logs with identity theft patterns.
        
        Args:
            n_samples: Number of samples to generate.
            
        Returns:
            DataFrame with synthetic transaction logs.
        """
        data = []
        
        for i in range(n_samples):
            # Determine if this is a legitimate or suspicious transaction
            is_theft = random.random() < 0.1  # 10% identity theft cases
            
            if is_theft:
                # Generate suspicious patterns
                login_hour = random.choice(self.user_patterns['suspicious_login_hours'])
                device_match = random.random() < 0.3  # 30% chance of known device
                location_match = random.random() < 0.2  # 20% chance of known location
                multiple_attempts = random.random() < 0.8  # 80% chance of multiple attempts
                session_duration = random.uniform(*self.user_patterns['suspicious_session_duration'])
                transaction_amount = random.uniform(*self.user_patterns['suspicious_transaction_amount'])
                time_since_last_login = random.uniform(0, 2)  # 0-2 hours
            else:
                # Generate normal patterns
                login_hour = random.choice(self.user_patterns['normal_login_hours'])
                device_match = random.random() < 0.85  # 85% chance of known device
                location_match = random.random() < 0.90  # 90% chance of known location
                multiple_attempts = random.random() < 0.1  # 10% chance of multiple attempts
                session_duration = random.uniform(*self.user_patterns['normal_session_duration'])
                transaction_amount = random.uniform(*self.user_patterns['normal_transaction_amount'])
                time_since_last_login = random.uniform(2, 48)  # 2-48 hours
            
            # Generate additional features
            device_id = random.choice(self.device_fingerprints)
            location = random.choice(self.locations)
            
            # Create synthetic user ID (hashed for privacy)
            user_id = self._hash_sensitive_data(f"user_{i}")
            
            # Generate browser and OS info
            browser_types = ['Chrome', 'Firefox', 'Safari', 'Edge']
            os_types = ['Windows', 'macOS', 'Linux', 'Android', 'iOS']
            
            browser = random.choice(browser_types)
            os = random.choice(os_types)
            
            # Generate transaction type
            transaction_types = ['login', 'transfer', 'purchase', 'withdrawal', 'deposit']
            transaction_type = random.choice(transaction_types)
            
            # Generate risk features
            ip_reputation_score = random.uniform(0, 1) if is_theft else random.uniform(0.7, 1)
            velocity_score = random.uniform(0.8, 1) if is_theft else random.uniform(0, 0.3)
            
            record = {
                'user_id': user_id,
                'timestamp': self.fake.date_time_between(start_date='-30d', end_date='now'),
                'login_hour': login_hour,
                'day_of_week': random.randint(0, 6),
                'device_id': device_id,
                'device_match': int(device_match),
                'browser': browser,
                'os': os,
                'location_city': location['city'],
                'location_country': location['country'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'location_match': int(location_match),
                'ip_address': self._hash_sensitive_data(f"ip_{i}"),
                'ip_reputation_score': ip_reputation_score,
                'session_duration': session_duration,
                'time_since_last_login': time_since_last_login,
                'multiple_attempts': int(multiple_attempts),
                'failed_login_attempts': random.randint(0, 5) if multiple_attempts else 0,
                'transaction_type': transaction_type,
                'transaction_amount': transaction_amount,
                'velocity_score': velocity_score,
                'account_age_days': random.randint(30, 3650),
                'previous_fraud_reports': random.randint(0, 3),
                'identity_theft': int(is_theft)
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def generate_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate graph-based features for entity relationships.
        
        Args:
            df: Transaction logs DataFrame.
            
        Returns:
            DataFrame with additional graph features.
        """
        # Device sharing patterns
        device_counts = df['device_id'].value_counts()
        df['device_sharing_count'] = df['device_id'].map(device_counts)
        
        # IP sharing patterns
        ip_counts = df['ip_address'].value_counts()
        df['ip_sharing_count'] = df['ip_address'].map(ip_counts)
        
        # Location-based features
        location_counts = df.groupby(['location_city', 'location_country']).size()
        df['location_frequency'] = df.apply(
            lambda x: location_counts.get((x['location_city'], x['location_country']), 0), 
            axis=1
        )
        
        # Time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['login_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['login_hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_train_test_split(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """Create time-aware train/validation/test splits.
        
        Args:
            df: Full dataset DataFrame.
            test_size: Proportion of data for testing.
            val_size: Proportion of data for validation.
            
        Returns:
            Dictionary with train/val/test splits.
        """
        # Sort by timestamp for time-aware splitting
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate split indices
        n_samples = len(df_sorted)
        test_start = int(n_samples * (1 - test_size))
        val_start = int(n_samples * (1 - test_size - val_size))
        
        # Create splits
        train_df = df_sorted[:val_start]
        val_df = df_sorted[val_start:test_start]
        test_df = df_sorted[test_start:]
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }


def generate_synthetic_data(n_samples: int = 10000) -> Dict[str, Any]:
    """Generate complete synthetic dataset for identity theft detection.
    
    Args:
        n_samples: Number of samples to generate.
        
    Returns:
        Dictionary containing train/val/test splits and metadata.
    """
    generator = IdentityTheftDataGenerator(seed=42)
    
    # Generate transaction logs
    df = generator.generate_transaction_logs(n_samples)
    
    # Add graph features
    df = generator.generate_graph_features(df)
    
    # Create splits
    splits = generator.create_train_test_split(df)
    
    # Prepare features and targets
    feature_columns = [
        'login_hour', 'day_of_week', 'device_match', 'location_match',
        'session_duration', 'time_since_last_login', 'multiple_attempts',
        'failed_login_attempts', 'transaction_amount', 'velocity_score',
        'account_age_days', 'previous_fraud_reports', 'ip_reputation_score',
        'device_sharing_count', 'ip_sharing_count', 'location_frequency',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]
    
    result = {}
    for split_name, split_df in splits.items():
        X = split_df[feature_columns]
        y = split_df['identity_theft']
        
        result[f'X_{split_name}'] = X
        result[f'y_{split_name}'] = y
        result[f'{split_name}_df'] = split_df
    
    result['feature_columns'] = feature_columns
    result['metadata'] = {
        'n_samples': n_samples,
        'n_features': len(feature_columns),
        'fraud_rate': df['identity_theft'].mean(),
        'generator_seed': 42
    }
    
    return result


if __name__ == "__main__":
    # Generate sample data for testing
    data = generate_synthetic_data(1000)
    print(f"Generated dataset with {data['metadata']['n_samples']} samples")
    print(f"Fraud rate: {data['metadata']['fraud_rate']:.2%}")
    print(f"Features: {data['metadata']['n_features']}")
    print(f"Train samples: {len(data['X_train'])}")
    print(f"Val samples: {len(data['X_val'])}")
    print(f"Test samples: {len(data['X_test'])}")
