# Databricks notebook source
from datetime import datetime
import mlflow

def read_data(data_path):
    return spark.read.parquet(data_path)

def apply_transformation_model(data, transformation_model):
    return transformation_model.transform(data)

def apply_inference_model(data, inference_model):
    return inference_model.transform(data)

def load_model_with_error_handling(model_run_id, model_type):
    try:
        model = mlflow.spark.load_model(f'runs:/{model_run_id}')
        print(f"Successfully loaded {model_type} model.")
        return model
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        # Handle error as needed
        raise e

# File paths and model run IDs
inference_data_path = "/mnt/silver_layer/transformed_data.parquet"
transformation_pipeline_run_id = '124c33a75d3b410a9d8c9cb03f2180fb/transformation'
ml_pipeline_run_id = '124c33a75d3b410a9d8c9cb03f2180fb/model'

# Read data
df_read = read_data(inference_data_path)

# Load transformation model
transformation_model = load_model_with_error_handling(transformation_pipeline_run_id, "transformation")

# Apply transformation
transformed_data = apply_transformation_model(df_read, transformation_model)

# Load inference model
inference_model = load_model_with_error_handling(ml_pipeline_run_id, "inference")

# Apply inference
predictions = apply_inference_model(transformed_data, inference_model)


current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
predictions.write.parquet(f"/mnt/gold_layer/predictions_{current_datetime}")

predictions.show(1)

