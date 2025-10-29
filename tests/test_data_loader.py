"""
Unit tests for DataLoader module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.data.data_loader import DataLoader, load_raw_data


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample transaction data."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Time': np.random.uniform(0, 172792, n_samples),
            'Amount': np.random.uniform(0, 1000, n_samples),
            'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
        }
        
        # Add V1-V28 features
        for i in range(1, 29):
            data[f'V{i}'] = np.random.randn(n_samples)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        
        # Cleanup
        Path(f.name).unlink()
    
    def test_init_with_valid_path(self, temp_csv_file):
        """Test initialization with valid file path."""
        loader = DataLoader(temp_csv_file)
        assert loader.data_path.exists()
        assert loader.df is None
    
    def test_init_with_invalid_path(self):
        """Test initialization with invalid file path."""
        with pytest.raises(FileNotFoundError):
            DataLoader("nonexistent_file.csv")
    
    def test_load_data(self, temp_csv_file):
        """Test data loading."""
        loader = DataLoader(temp_csv_file)
        df = loader.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert loader.df is not None
    
    def test_load_data_shape(self, temp_csv_file):
        """Test loaded data has correct shape."""
        loader = DataLoader(temp_csv_file)
        df = loader.load_data()
        
        assert df.shape[1] == 31  # Time, V1-V28, Amount, Class
    
    def test_get_data_summary(self, temp_csv_file):
        """Test data summary generation."""
        loader = DataLoader(temp_csv_file)
        loader.load_data()
        
        summary = loader.get_data_summary()
        
        assert 'n_rows' in summary
        assert 'n_columns' in summary
        assert 'class_distribution' in summary
        assert 'fraud_rate' in summary
        assert summary['n_columns'] == 31
    
    def test_validate_schema_success(self, temp_csv_file):
        """Test schema validation with correct columns."""
        loader = DataLoader(temp_csv_file)
        loader.load_data()
        
        expected_cols = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
        assert loader.validate_schema(expected_cols)
    
    def test_validate_schema_missing_columns(self, temp_csv_file):
        """Test schema validation with missing columns."""
        loader = DataLoader(temp_csv_file)
        loader.load_data()
        
        expected_cols = ['Time', 'Amount', 'Class', 'MissingColumn'] + [f'V{i}' for i in range(1, 29)]
        
        with pytest.raises(ValueError):
            loader.validate_schema(expected_cols)
    
    def test_convenience_function(self, temp_csv_file):
        """Test load_raw_data convenience function."""
        df = load_raw_data(temp_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestDataLoaderEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write only header
            f.write('Time,Amount,Class\n')
            temp_path = f.name
        
        loader = DataLoader(temp_path)
        
        with pytest.raises(ValueError, match="empty"):
            loader.load_data()
        
        Path(temp_path).unlink()
    
    def test_get_summary_without_loading(self):
        """Test get_data_summary before loading data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('Time,Amount,Class\n0,100,0\n')
            temp_path = f.name
        
        loader = DataLoader(temp_path)
        
        with pytest.raises(ValueError):
            loader.get_data_summary()
        
        Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
