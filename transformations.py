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


