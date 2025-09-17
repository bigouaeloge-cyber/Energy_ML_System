# debug_helper.py
import sys
import traceback
from pathlib import Path

def quick_test():
    """Test rapide de tous les imports"""
    modules = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 
        'streamlit', 'plotly', 'requests'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    # Test imports personnalisés
    sys.path.append('src')
    try:
        from data_processing import DataProcessor
        print("✅ DataProcessor")
    except Exception as e:
        print(f"❌ DataProcessor: {e}")

if __name__ == '__main__':
    quick_test()