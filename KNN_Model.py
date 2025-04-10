Python 3.11.5 (v3.11.5:cce6ba91b3, Aug 24 2023, 10:50:31) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
pip install databricks-connect
databricks configure --token
from pyspark.sql import SparkSession
import os
import pandas as pd

# Set Databricks workspace URL and token
df1 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/julianvazquez171@gmail.com/clean_aircraft_data-1.csv")
dbutils_token = "julianvazquez171@gmail.com/?o=2426367061272425#notebook/1528223172663301"
workspace_url = "https://community.cloud.databricks.com/?o=2426367061272425#notebook/1528223172663301/command/1528223172663302>"

# Configure Databricks Connect
os.environ['DATABRICKS_TOKEN'] = dbutils_token
os.environ['DATABRICKS_HOST'] = workspace_url

# Set other Spark configurations
spark = SparkSession.builder \
    .appName("AircraftPricePrediction") \
    .config("spark.databricks.workspaceUrl", workspace_url) \
    .getOrCreate()

df1 = pd.read_csv("/dbfs/FileStore/shared_uploads/julianvazquez171@gmail.com/clean_aircraft_data-2.csv")

# Verify the connection
print(spark.range(5).collect())

from pyspark.sql import SparkSession
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

file_path = 'dbfs:/FileStore/tables/cleaned_aircraft_data.csv'
data = pd.read_csv(file_path)

X = data[['Number', 'Year', 'Total Hours']]  # Features
y = data['Price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a k-NN model
knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors
knn_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = knn_model.predict(X_test)

# Random sample of 10 elements
sample_data = X_test.sample(n=10, random_state=42)
sample_actual_prices = y_test.loc[sample_data.index]

# Make predictions for the sample
sample_predictions = knn_model.predict(sample_data)
sample_results_df = pd.DataFrame({
    'Model': ['k-NN'] * 10,
    'Number': sample_data['Number'].values,
    'Actual Price': sample_actual_prices.values,
    'Predicted Price': sample_predictions,
    'Difference': sample_actual_prices.values - sample_predictions
})

# Print the DataFrame for the random sample
print("\nRandom Sample Results Table:")
print(sample_results_df)
