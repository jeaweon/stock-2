"""Unit tests for stock data loader."""
import unittest
from pathlib import Path
import pandas as pd
from src.data_loader import load_stock_data, get_cache_path


class TestDataLoader(unittest.TestCase):
    def test_load_stock_data_columns_and_types(self):
        # Test loading AAPL for a short period to verify format quickly
        df = load_stock_data("AAPL", period="5d", interval="1h")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            self.assertIn(col, df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df.index))

    def test_cache_creation(self):
        cache_path = get_cache_path("AAPL", period="5d", interval="1h")
        self.assertTrue(cache_path.exists())


if __name__ == "__main__":
    unittest.main()
