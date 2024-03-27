# Databricks notebook source
from pyspark.sql import SparkSession

# Define the mount point and connection details
mount_point = "/mnt/bronze_layer"
storage_account_name = "storageiscs"
container_name = "bronze"

mounts = dbutils.fs.mounts()
if not any(mount.mountPoint == mount_point for mount in mounts):
  dbutils.fs.mount(
    source="wasbs://"+container_name+"@"+storage_account_name+".blob.core.windows.net",
    mount_point=mount_point,
    extra_configs={
      "fs.azure.account.key."+storage_account_name+".blob.core.windows.net": dbutils.secrets.get('DatalakeScope','StorageSecretKey')
    }
  )


# COMMAND ----------

# Define the mount point and connection details
mount_point = "/mnt/silver_layer"
storage_account_name = "storageiscs"
container_name = "silver"

mounts = dbutils.fs.mounts()
if not any(mount.mountPoint == mount_point for mount in mounts):
  dbutils.fs.mount(
    source="wasbs://"+container_name+"@"+storage_account_name+".blob.core.windows.net",
    mount_point=mount_point,
    extra_configs={
      "fs.azure.account.key."+storage_account_name+".blob.core.windows.net": dbutils.secrets.get('DatalakeScope','StorageSecretKey')
    }
  )

# COMMAND ----------

# Define the mount point and connection details
mount_point = "/mnt/gold_layer"
storage_account_name = "storageiscs"
container_name = "gold"

mounts = dbutils.fs.mounts()
if not any(mount.mountPoint == mount_point for mount in mounts):
  dbutils.fs.mount(
    source="wasbs://"+container_name+"@"+storage_account_name+".blob.core.windows.net",
    mount_point=mount_point,
    extra_configs={
      "fs.azure.account.key."+storage_account_name+".blob.core.windows.net": dbutils.secrets.get('DatalakeScope','StorageSecretKey')
    }
  )

# COMMAND ----------

dbutils.secrets.listScopes()
