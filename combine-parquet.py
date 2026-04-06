import pandas as pd
import glob

# # Get all parquet files in a directory
# file_list = glob.glob("data/*/all_good_points.parquet")

# # Read and concatenate all files
# df = pd.concat([pd.read_parquet(file) for file in file_list])

# # Save the combined dataframe to a new Parquet file
# df.to_parquet("combined-good-points.parquet", engine="pyarrow")


folders = [
    "data/2025-12-28-16-36-dev",
]

file_list = [f"{folder}/all_good_points.parquet" for folder in folders]

# # Also include your existing combined parquet file
# file_list.append("data/25-08-2025-combined/combined-good-points.parquet")

print("Files being combined:")
for f in file_list:
    print(f)

df = pd.concat([pd.read_parquet(file) for file in file_list], ignore_index=True)

df.to_parquet("all_good_points_high_theta_over_pi4.parquet", engine="pyarrow")