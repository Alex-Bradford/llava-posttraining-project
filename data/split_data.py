import pandas as pd
from sklearn.model_selection import train_test_split

# Path to your source parquet file
input_path = "data/lrv_chart.parquet"

# Load the full dataset
df = pd.read_parquet(input_path)

# Optional: drop 'data_source' if you don't need it
df = df.drop(columns=["data_source"], errors="ignore")

# Split into train and test sets (e.g. 90% train, 10% test)
train_df, test_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

# # Save the train and test sets to separate parquet files
# train_df.to_parquet("data/train.parquet", index=False)
# test_df.to_parquet("data/test.parquet", index=False)

print("✅ Successfully split and saved train/test datasets.")