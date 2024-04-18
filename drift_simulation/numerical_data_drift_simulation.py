# Databricks notebook source
# MAGIC %run ./Data_pipelines/transformations

# COMMAND ----------

df = spark.read.format("csv").option("header", "true").load("/mnt/inference_layer/batch_2.csv")

column_types = {
    "Model": StringType(),
    "Price": DoubleType(),
    "Age_08_04": IntegerType(),
    "Mfg_Month": IntegerType(),
    "Mfg_Year": IntegerType(),
    "KM": IntegerType(),
    "Fuel_Type": StringType(),
    "HP": IntegerType(),
    "Met_Color": StringType(),
    "Color": StringType(),
    "Automatic": StringType(),
    "cc": IntegerType(),
    "Doors": IntegerType(),
    "Gears": IntegerType(),
    "Quarterly_Tax": IntegerType(),
    "Weight": IntegerType(),
    "Mfr_Guarantee": StringType(),
    "BOVAG_Guarantee": StringType(),
    "Guarantee_Period": IntegerType(),
    "ABS": StringType(),
    "Airbag_1": StringType(),
    "Airbag_2": StringType(),
    "Airco": StringType(),
    "Automatic_airco": StringType(),
    "Boardcomputer": StringType(),
    "CD_Player": StringType(),
    "Central_Lock": StringType(),
    "Powered_Windows": StringType(),
    "Power_Steering": StringType(),
    "Radio": StringType(),
    "Mistlamps": StringType(),
    "Sport_Model": StringType(),
    "Backseat_Divider": StringType(),
    "Metallic_Rim": StringType(),
    "Radio_cassette": StringType(),
    "Tow_Bar": StringType(),
    "Cylinders": IntegerType()
}

convert_dtype_inference = DataFrameTransformer()
df = convert_dtype_inference.remove_unnecessary_columns(df, columns_to_remove)
df = convert_dtype_inference.convert_data_types(df, column_types)

from pyspark.sql.types import StringType, DoubleType, IntegerType
from pyspark.sql import DataFrame
from typing import Tuple

def separate_columns(df: DataFrame, target_column_name: str) -> Tuple[list, list, str]:
    numerical_columns = []
    categorical_columns = []
    target_column = None

    for column, dtype in df.dtypes:
        if column == target_column_name:
            target_column = column
        elif dtype == "string":
            categorical_columns.append(column)
        else:
            numerical_columns.append(column)

    return numerical_columns, categorical_columns, target_column

# Example usage:
# Assuming df is your DataFrame and "Price" is your target column name
numerical_cols, categorical_cols, target_col = separate_columns(df, "Price")
print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)
print("Target Column:", target_col)


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, stddev, randn

# Specify numerical columns
numerical_columns = ['Age_08_04', 'Mfg_Month', 'Mfg_Year', 'KM', 'HP', 'cc', 'Doors', 'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight', 'Guarantee_Period', 'Car_Age', 'Mileage']

# Calculate standard deviation for each numerical column
std_devs = df.select(*(stddev(col(c)).alias(c) for c in numerical_columns)).collect()[0]

# Define drift factors based on standard deviation
drift_factors = {col_name: std_dev * (0.1 + (std_dev / 100)) for col_name, std_dev in zip(numerical_columns, std_devs)}

# Introduce drift
for column, factor in drift_factors.items():
    df = df.withColumn(column, col(column) + (randn() * factor))

# Show DataFrame with drift
display(df)
