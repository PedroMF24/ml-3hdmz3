# %%
import os

import glob
import pandas as pd

from tqdm import tqdm


from src.utils.data import save_parameters_fortran_file, parameter_columns

# %%

os.makedirs("plots", exist_ok=True)

# %%

scans = [
    "analysis_theta_results"
]
# all_good_points_below_59 # reader_check_HT
for scan in tqdm(scans):
    for file_path in tqdm(glob.glob(f"dados/{scan}/analysis_theta_results.parquet")): #   combined-good-points.parquet all_good_points.parquet
        loaded_points = pd.read_parquet(file_path)
        save_parameters_fortran_file(
            loaded_points[parameter_columns], f"dados/{scan}.dat", True
        )

# scan = "tests-combined" # "05-09-2025-combined"

# file_path = f"data/{scan}/combined-good-points.parquet"

# loaded_points = pd.read_parquet(file_path)

# print(len(loaded_points))

# save_parameters_fortran_file(
#             loaded_points[parameter_columns], f"dados/{scan}.dat", True
#         )

# %%
