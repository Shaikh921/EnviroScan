"""
Unit tests for configuration
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR,
    ALL_FEATURES, POLLUTION_SOURCES,
    get_aqi_status, ensure_directories
)


class TestConfiguration:
    """Test configuration settings"""
    
    def test_project_root_exists(self):
        """Test that project root is valid"""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()
    
    def test_feature_lists(self):
        """Test feature list completeness"""
        assert len(ALL_FEATURES) == 14
        assert 'pm25' in ALL_FEATURES
        assert 'Temperature' in ALL_FEATURES
        assert 'dist_to_road' in ALL_FEATURES
    
    def test_pollution_sources(self):
        """Test pollution source categories"""
        assert len(POLLUTION_SOURCES) == 5
        assert 'Industrial' in POLLUTION_SOURCES
        assert 'Vehicular' in POLLUTION_SOURCES
        assert 'Agricultural' in POLLUTION_SOURCES
        assert 'Burning' in POLLUTION_SOURCES
        assert 'Natural' in POLLUTION_SOURCES


class TestAQIStatus:
    """Test AQI status function"""
    
    def test_aqi_good(self):
        """Test Good AQI range"""
        status, color, emoji = get_aqi_status(30)
        assert status == 'Good'
        assert color == 'green'
        assert emoji == '🟢'
    
    def test_aqi_moderate(self):
        """Test Moderate AQI range"""
        status, color, emoji = get_aqi_status(75)
        assert status == 'Moderate'
        assert color == 'yellow'
        assert emoji == '🟡'
    
    def test_aqi_poor(self):
        """Test Poor AQI range"""
        status, color, emoji = get_aqi_status(150)
        assert status == 'Poor'
        assert color == 'orange'
        assert emoji == '🟠'
    
    def test_aqi_very_poor(self):
        """Test Very Poor AQI range"""
        status, color, emoji = get_aqi_status(250)
        assert status == 'Very Poor'
        assert color == 'red'
        assert emoji == '🔴'
    
    def test_aqi_hazardous(self):
        """Test Hazardous AQI range"""
        status, color, emoji = get_aqi_status(350)
        assert status == 'Hazardous'
        assert color == 'purple'
        assert emoji == '🟣'
    
    def test_aqi_boundary_values(self):
        """Test AQI boundary values"""
        # Test exact boundaries
        status, _, _ = get_aqi_status(50)
        assert status == 'Good'
        
        status, _, _ = get_aqi_status(51)
        assert status == 'Moderate'
        
        status, _, _ = get_aqi_status(100)
        assert status == 'Moderate'
        
        status, _, _ = get_aqi_status(101)
        assert status == 'Poor'


class TestDirectoryManagement:
    """Test directory management functions"""
    
    def test_ensure_directories(self):
        """Test directory creation"""
        # This should not raise any errors
        ensure_directories()
        
        # Check that key directories exist
        assert DATA_DIR.exists()
        assert MODELS_DIR.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
