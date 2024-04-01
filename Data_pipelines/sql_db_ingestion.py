# Databricks notebook source
# Import required libraries
import pyodbc 
from pyspark.sql import SparkSession

# Create dataframe by querying Azure SQL Database
df = spark.read.format("jdbc").option("url", dbutils.secrets.get('dbscope','dbKey')).option("dbtable", "SalesLT.Customer").load()

# COMMAND ----------

df.show(10)

# COMMAND ----------

dbutils.secrets.listScopes()

# COMMAND ----------

print(dbutils.secrets.get('dbscope','dbKey'))

# COMMAND ----------

# Define the query
query = "SELECT * FROM SalesLT.Customer"

# Read the data from the SQL database based on the query
data = spark.read.format("jdbc").option("url", dbutils.secrets.get('dbscope','dbKey')).option("dbtable", f"({query}) AS subquery").load()

# Print the data
data.show()

# COMMAND ----------

data.write.format("parquet").mode("overwrite").save("/mnt/bronze_layer/customer_data")

# COMMAND ----------

dbutils.fs.mounts()

# COMMAND ----------


