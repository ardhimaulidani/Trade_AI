import pandas as pd

# Path to your Parquet file
parquet_path = "database/parquet/event_trades.parquet"

# Read the Parquet file
df = pd.read_parquet(parquet_path)
# Convert timezone-aware datetimes → timezone-naive
for col in df.select_dtypes(['datetimetz']).columns:
    df[col] = df[col].dt.tz_convert(None)

# Save it as Excel
excel_path = "tpo_feature.xlsx"
df.to_excel(excel_path, index=False)

print(f"✅ Converted successfully: {excel_path}")
