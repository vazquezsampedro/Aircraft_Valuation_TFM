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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# Start a Spark session
spark = SparkSession.builder.appName("AircraftData").getOrCreate()

# Path to the CSV file on your local file system
local_file_path = "/path/to/your/local/cleaned_aircraft_data.csv"

# Path to the directory on Databricks where you want to save the file
dbfs_file_path = "dbfs:/FileStore/tables/cleaned_aircraft_data.csv"

# Copy the CSV file to the Databricks DBFS file system
dbutils.fs.cp("file:" + local_file_path, dbfs_file_path)

# Load the CSV file into a Spark DataFrame
spark_df = spark.read.option("header", "true").csv(dbfs_file_path)

# Show the first records of the DataFrame
spark_df.show()

# Load data from the CSV file
file_path = 'dbfs:/FileStore/tables/cleaned_aircraft_data.csv'
data = pd.read_csv(file_path)

# Split the data into training and testing sets
X = data[['Number', 'Year', 'Total Hours']]  # Features
y = data['Price']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
plt.scatter(X_test['Total Hours'], y_test, color='black', label='Actual Price')
plt.scatter(X_test['Total Hours'], predictions, color='blue', label='Predicted Price')
plt.xlabel('Total Hours')
plt.ylabel('Price')
plt.legend()
plt.show()
