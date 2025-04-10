Python 3.11.5 (v3.11.5:cce6ba91b3, Aug 24 2023, 10:50:31) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> pip install databricks-connect
... databricks configure --token
... from pyspark.sql import SparkSession
... import os
... import pandas as pd
... 
... # Set your Databricks workspace URL and token
... df1 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/julianvazquez171@gmail.com/clean_aircraft_data-1.csv")
... dbutils_token = "ulianvazquez171@gmail.com/?o=2426367061272425#notebook/1528223172663301"
... workspace_url = "https://community.cloud.databricks.com/?o=2426367061272425#notebook/1528223172663301/command/1528223172663302>"
... 
... # Configure Databricks Connect
... os.environ['DATABRICKS_TOKEN'] = dbutils_token
... os.environ['DATABRICKS_HOST'] = workspace_url
... 
... # Set other Spark configurations
... spark = SparkSession.builder \
...     .appName("AircraftPricePrediction") \
...     .config("spark.databricks.workspaceUrl", workspace_url) \
...     .getOrCreate()
... 
... df1 = pd.read_csv("/dbfs/FileStore/shared_uploads/julianvazquez171@gmail.com/clean_aircraft_data-2.csv")
... 
... # Verify the connection
... print(spark.range(5).collect())
