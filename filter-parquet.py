import pandas as pd
import numpy as np

# file_path = "data/2025-08-16-10-56-dev/"

file_path = "data/2025-10-07-18-28-dev/"

# "2025-10-02-01-00-dev/" # 25-08-2025-combined/

file_name = file_path + "all_good_points.parquet"

# file_name = file_path + "inspect_points_reader.csv"

from src.utils.data import (
    all_columns,
    HT_columns,
    MO_columns,
    MO_id_column,
)

all_columns = all_columns + HT_columns + MO_columns + MO_id_column

new_df = pd.read_parquet(file_name)
# new_df = pd.read_csv(file_name, header=0, sep=",", dtype=np.float64)
# new_df = pd.read_csv(file_name, header=0, sep=",")
# df_filtered = new_df
print(new_df.columns.tolist())

# Keep only rows where both columns are not null
# df_filtered = new_df[
#     (pd.to_numeric(new_df["BR_h_to_H1H1"], errors="coerce") > 0) &
#     (pd.to_numeric(new_df["BR_h_to_A1A1"], errors="coerce") > 0)
# ]

new_df["OmegaT"] = new_df["Omega_1"] + new_df["Omega_2"]

# threshold = 0.1e-10
# df_filtered = new_df[new_df['MH10']<63]
# df_filtered = new_df[new_df['MH10'] > 77.5]

mask = (
    (new_df["OmegaT"] > 0.1164) &
    (new_df["OmegaT"] < 0.1236) & 
    (new_df["MH10"] < 100)
)


df_filtered = new_df[mask]

# print(df_filtered["OmegaT"].min(), df_filtered["OmegaT"].max())
# print(df_filtered["MH10"].min(), df_filtered["MH10"].max())

df_filtered = df_filtered.drop(columns=["OmegaT"])



# Omega_lower: 0.1118 # 0.1164
# Omega_upper: 0.1278 # 0.1236


# D_n = new_df["MH20"] - new_df["MH10"]
# D_c = new_df["mC1"] - new_df["MH10"]
# d_c = new_df["mC2"] - new_df["mC1"]

# tol_n = 5   # for D_n ~ 50
# tol_c = 5   # for D_c ~ 60
# tol_dc = 2  # for d_c ~ 10

# # Scenario B
# mask_B = (
#     (np.abs(D_n - 50) <= tol_n) &
#     (np.abs(D_c - 60) <= tol_c) &
#     (np.abs(d_c - 10) <= tol_dc)
# )

# tol_n = 2   # for D_n ~ 50
# tol_c = 5   # for D_c ~ 60
# tol_dc = 0.5  # for d_c ~ 10

# mask_C = (
#     (np.abs(D_n - 10) <= tol_n) &
#     (np.abs(D_c - 50) <= tol_c) &
#     (np.abs(d_c - 1) <= tol_dc)
# )


# df_filtered = new_df[new_df[mask_B]]



# df_filtered = new_df[new_df['dd_H1_SI_CS']>threshold]
# df_filtered =  df_filtered[df_filtered['MH10']>77]
# output_file = file_path + "filtered.parquet"

# output_file = file_path + "combined-filtered.parquet"

output_file = file_path + "all_good_points.parquet" # filtered_

df_filtered.to_parquet(output_file, engine="pyarrow", index=False)

# new_df.to_parquet(output_file, engine="pyarrow", index=False)
