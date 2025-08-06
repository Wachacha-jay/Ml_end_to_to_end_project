import sys
import os
sys.path.append('.')

try:
    from src.components.data_transformation import DataTransformation
    from src.components.data_ingestion import DataIngestion
    
    print("Testing data transformation...")
    
    # Test data ingestion first
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()
    print(f"Data ingestion successful. Train: {train_data}, Test: {test_data}")
    
    # Test data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    print(f"Data transformation successful!")
    print(f"Train array shape: {train_arr.shape}")
    print(f"Test array shape: {test_arr.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 