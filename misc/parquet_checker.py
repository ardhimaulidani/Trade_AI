import pandas as pd
import pyarrow.parquet as pq

# Get schema without loading data
parquet_file = pq.ParquetFile('database/parquet/btc_tpo_features.parquet')
df = pd.read_parquet('database/parquet/btc_tpo_features.parquet')

print("Schema:")
print(parquet_file.schema)
print("\nColumn names:")
print(parquet_file.schema.names)
print("Tables: ")
print(df.tail().T.iloc[-24:])