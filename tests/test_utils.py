"""
Unit tests for utility functions
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import (
    remove_missing_values,
    remove_duplicates,
    validate_input_data,
    get_source_color,
    add_noise_to_features
)


class TestDataCleaning:
    """Test data cleaning functions"""
    
    def test_remove_missing_values_drop(self):
        """Test dropping missing values"""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, np.nan, 7, 8]
        })
        
        result = remove_missing_values(df, strategy='drop')
        assert len(result) == 2
        assert result.isnull().sum().sum() == 0
    
    def test_remove_duplicates(self):
        """Test removing duplicate rows"""
        df = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': [4, 5, 5, 6]
        })
        
        result = remove_duplicates(df)
        assert len(result) == 3
    
    def test_remove_duplicates_subset(self):
        """Test removing duplicates based on subset"""
        df = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': [4, 5, 6, 7]
        })
        
        result = remove_duplicates(df, subset=['a'])
        assert len(result) == 3


class TestModelUtilities:
    """Test model-related utility functions"""
    
    def test_validate_input_data_valid(self):
        """Test validation with valid data"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        required = ['feature1', 'feature2']
        assert validate_input_data(df, required) == True
    
    def test_validate_input_data_missing_features(self):
        """Test validation with missing features"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3]
        })
        
        required = ['feature1', 'feature2']
        with pytest.raises(ValueError, match="Missing required features"):
            validate_input_data(df, required)
    
    def test_validate_input_data_nan_values(self):
        """Test validation with NaN values"""
        df = pd.DataFrame({
            'feature1': [1, np.nan, 3],
            'feature2': [4, 5, 6]
        })
        
        required = ['feature1', 'feature2']
        with pytest.raises(ValueError, match="NaN values"):
            validate_input_data(df, required)
    
    def test_add_noise_to_features(self):
        """Test adding noise to features"""
        df = pd.DataFrame({
            'feature1': [10.0] * 100,
            'feature2': [20.0] * 100
        })
        
        result = add_noise_to_features(df, noise_level=0.1)
        
        # Check that noise was added (values should differ)
        assert not np.allclose(result['feature1'].values, df['feature1'].values)
        assert not np.allclose(result['feature2'].values, df['feature2'].values)
        
        # Check that mean is approximately preserved
        assert np.abs(result['feature1'].mean() - 10.0) < 1.0
        assert np.abs(result['feature2'].mean() - 20.0) < 1.0


class TestVisualizationUtilities:
    """Test visualization utility functions"""
    
    def test_get_source_color(self):
        """Test pollution source color mapping"""
        assert get_source_color("Industrial") == "red"
        assert get_source_color("Vehicular") == "blue"
        assert get_source_color("Agricultural") == "green"
        assert get_source_color("Burning") == "orange"
        assert get_source_color("Natural") == "purple"
        assert get_source_color("Unknown") == "gray"


class TestAPIUtilities:
    """Test API-related functions"""
    
    def test_fetch_weather_data_invalid_key(self):
        """Test weather API with invalid key"""
        from utils import fetch_weather_data
        
        result = fetch_weather_data(28.6139, 77.2090, "invalid_key", retries=1)
        assert result is None
    
    def test_fetch_pollution_data_invalid_key(self):
        """Test pollution API with invalid key"""
        from utils import fetch_pollution_data
        
        result = fetch_pollution_data(28.6139, 77.2090, "invalid_key", retries=1)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
