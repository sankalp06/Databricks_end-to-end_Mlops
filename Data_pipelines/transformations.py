# Databricks notebook source
from pyspark.sql.functions import expr
from pyspark.sql.types import StringType, IntegerType, DoubleType
from datetime import datetime

class DataFrameTransformer:
    def __init__(self):
        pass

    def convert_data_types(self, data_df, column_types=None):
        if column_types:
            for column, data_type in column_types.items():
                data_df = data_df.withColumn(column, data_df[column].cast(data_type))
        return data_df

    def apply_feature_engineering(self, data_df, transformations=None):
        if transformations and data_df is not None:
            for new_column, expression in transformations.items():
                try:
                    data_df = data_df.withColumn(new_column, expr(expression))
                except Exception as e:
                    print(f"Error applying transformation for {new_column}: {e}")
        return data_df

    def remove_unnecessary_columns(self, data_df, columns_to_remove=None):
        if columns_to_remove:
            data_df = data_df.drop(*columns_to_remove)
        return data_df

    def handle_outliers(self, data_df, numerical_columns=None, lower_bound=None, upper_bound=None):
        if numerical_columns and lower_bound is not None and upper_bound is not None:
            for column in numerical_columns:
                q1, q3 = data_df.approxQuantile(column, [0.25, 0.75], 0.01)
                iqr = q3 - q1
                lower_bound_val = q1 - 1.5 * iqr
                upper_bound_val = q3 + 1.5 * iqr
                data_df = data_df.withColumn(column, 
                                             expr(f"CASE WHEN ({column} < {lower_bound_val}) OR ({column} > {upper_bound_val}) THEN NULL ELSE {column} END"))
        return data_df



# COMMAND ----------

transformer = DataFrameTransformer()

# Read data
data_df = spark.read.format("csv").option("header", "true").load("/mnt/bronze_layer/ToyotaCorolla.csv")

# Define column types to convert
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

# Define columns to remove
columns_to_remove = ["Id"]

# Define feature engineering transformations
transformations = {
    "Car_Age": f"{datetime.now().year} - Mfg_Year",
    "Mileage": "KM / Car_Age"
}

# Define numerical columns list
numerical_columns = [
    "Age_08_04", "KM", "HP", "cc", "Doors", "Gears",
    "Quarterly_Tax", "Weight", "Guarantee_Period", "Mfg_Year", "Price"
]

# Apply transformations
data_df = transformer.convert_data_types(data_df, column_types)
data_df = transformer.apply_feature_engineering(data_df, transformations)
data_df = transformer.remove_unnecessary_columns(data_df, columns_to_remove)

# Show transformed data
data_df.show(1)


# COMMAND ----------

# Save the DataFrame to Parquet format
data_df.write.mode("overwrite").parquet("/mnt/silver_layer/transformed_data.parquet")

# COMMAND ----------

# Read the Parquet file
parquet_file_path = "/mnt/silver_layer/transformed_data.parquet"
df = spark.read.parquet(parquet_file_path)
