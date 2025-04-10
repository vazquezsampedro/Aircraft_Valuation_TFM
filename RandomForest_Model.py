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

pip install pandas scikit-learn
# Import the dbutils library
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# Load data from the CSV file
file_path = 'bfs:/FileStore/tables/cleaned_aircraft_data.csv'
data = pd.read_csv(file_path)

# Split the data into training and testing sets
X = data[['Number', 'Year', 'Total Hours']]  # Features
y = data['Price']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
plt.scatter(X_test['Total Hours'], y_test, color='black', label='Actual Price')
plt.scatter(X_test['Total Hours'], predictions, color='green', label='Predicted Price (Random Forest)')
plt.xlabel('Total Hours')
plt.ylabel('Price')
plt.legend()
plt.show()

# Get a sample of the first 10 aircraft
sample_data = X_test.head(10)
sample_actual_prices = y_test.head(10)
sample_predictions = model.predict(sample_data)

# Create a DataFrame to visualize the results
results_df = pd.DataFrame({
    'Actual Price': sample_actual_prices.values,
    'Predicted Price': sample_predictions,
    'Difference': sample_actual_prices.values - sample_predictions
})


# Get a sample of the first 10 aircraft
sample_data = X_test.head(10)
sample_actual_prices = y_test.head(10)

# Make predictions for the sample
sample_predictions = model.predict(sample_data)

# Calculate the difference in percentage
percentage_difference = ((sample_actual_prices.values - sample_predictions) / sample_actual_prices.values) * 100
extended_sample_data = X_test.head(10)
extended_sample_actual_prices = y_test.head(10)

extended_sample_predictions = model.predict(extended_sample_data)

extended_results_df = pd.DataFrame({
    'Model': ['RandomForestRegressor'] * 10,  # Indica el modelo utilizado
    'Number': extended_sample_data['Number'].values,
    'Actual Price': extended_sample_actual_prices.values,
    'Predicted Price': extended_sample_predictions,
    'Difference': extended_sample_actual_prices.values - extended_sample_predictions
})

print(extended_results_df)


extended_results_df = pd.DataFrame({
    'Model': ['RandomForestRegressor'] * 10,
    'Number': extended_sample_data['Number'].values,
    'Actual Price (4 years ago)': extended_sample_actual_prices.values,
    'Predicted Price': extended_sample_predictions,
    'Difference': extended_sample_actual_prices.values - extended_sample_predictions
})
print(extended_results_df)
turboprop_data = data[data.iloc[:, 3] == 'TURBOPROP'].sample(n=30, random_state=42)

# Select features for prediction (adjust based on actual columns in your data)
turboprop_features_for_prediction = turboprop_data[['Number', 'Year', 'Total Hours']]

turboprop_predictions = model.predict(turboprop_features_for_prediction)

# Create a DataFrame to visualize the results for Turboprop aircraft
turboprop_results_df = pd.DataFrame({
    'Number': turboprop_data['Number'].values,
    'Actual Price (4 years ago)': turboprop_data['Price'].values,
    'Predicted Price': turboprop_predictions,
    'Difference': turboprop_data['Price'].values - turboprop_predictions
})

# Print the DataFrame for Turboprop aircraft results
print(turboprop_results_df)
