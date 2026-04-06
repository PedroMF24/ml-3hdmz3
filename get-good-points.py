# %%
import os

import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
import numpy as np

from tqdm import tqdm


from src.utils.data import save_parameters_fortran_file, parameter_columns

# %%

os.makedirs("plots", exist_ok=True)

# %%

# Used for comparison in C
NUMERICAL_INF = np.inf
# Used as infinite penalty
NUMERICAL_INF_LOG = np.log(np.finfo(np.float64).max) + 1
EPS = np.finfo(np.float64).eps


scans = [
    "2026-02-23-17-54-dev",
]

# mode = "benchmark:B"
mode = "general"

constraints_bounds = yaml.safe_load(open("configs/constraints-bounds.yml", "r"))
for k, v in constraints_bounds.items():
    constraints_bounds[k] = eval(v) if isinstance(v, str) else v


parameter_bounds = yaml.safe_load(open("configs/parameter-bounds.yml", "r"))
for p, bs in parameter_bounds.items():
    for bk, bv in bs.items():
        parameter_bounds[p][bk] = eval(bv) if isinstance(bv, str) else bv

defaults = yaml.safe_load(open("configs/defaults.yml", "r"))
if os.path.isfile("defaults-local.yml"):
    defaults_local = yaml.safe_load(open("defaults-local.yml", "r"))
    if defaults_local:
        for k, v in defaults_local.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    defaults[k][kk] = vv
            else:
                defaults[k] = v

def collect_files(input_files_list, output_file):
    writer = None
    for input_file in input_files_list:
        input_file_df = pd.read_csv(input_file)
        table = pa.Table.from_pandas(input_file_df)
        if writer == None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
    writer.close()

def write_reader_parquet(defaults, results, output_path, input_file_name):
    all_good_points_files = glob.glob(os.path.join(output_path, "*", input_file_name))

    if defaults["collect_reader_seeds"]:
        table = pa.Table.from_pandas(results)
        pq.write_table(table, f"{output_path}/{input_file_name}")
        print(f"Parquet file successfully written: {output_path}/{input_file_name}")


def reverse_map_to_box_space(df):
    box_df = pd.DataFrame(dtype=np.float64)

    # MH125 is fixed at 1.0 box
    box_df["MH125_box"] = 1.0

    # Linear transform
    box_df["MH10_box"] = (df["MH10"] - parameter_bounds["MH10"]["low"]) / (
        parameter_bounds["MH10"]["up"] - parameter_bounds["MH10"]["low"]
    )

    # Conditional MH20 reverse
    mask_MH20 = df["MH20"] <= df["MH10"]
    box_df["MH20_box"] = np.where(
        mask_MH20,
        (df["MH20"] - df["MH10"]) / (parameter_bounds["MH20"]["up"] - df["MH10"]),
        (df["MH20"] - parameter_bounds["MH20"]["low"]) / (parameter_bounds["MH20"]["up"] - parameter_bounds["MH20"]["low"])
    )

    # mC1 conditional reverse
    mask_mC1 = df["mC1"] <= df["MH10"]
    box_df["mC1_box"] = np.where(
        mask_mC1,
        (df["mC1"] - df["MH10"] - 1e-3) / (parameter_bounds["mC1"]["up"] - df["MH10"]),
        (df["mC1"] - parameter_bounds["mC1"]["low"]) / (parameter_bounds["mC1"]["up"] - parameter_bounds["mC1"]["low"])
    )

    # mC2 conditional reverse
    mask_mC2 = df["mC2"] <= df["mC1"]
    box_df["mC2_box"] = np.where(
        mask_mC2,
        (df["mC2"] - df["mC1"] - 1e-3) / (parameter_bounds["mC2"]["up"] - df["mC1"]),
        (df["mC2"] - parameter_bounds["mC2"]["low"]) / (parameter_bounds["mC2"]["up"] - parameter_bounds["mC2"]["low"])
    )

    # Linear: theta, g1, g2
    for param in ["theta", "g1", "g2"]:
        box_df[param + "_box"] = (df[param] - parameter_bounds[param]["low"]) / (
            parameter_bounds[param]["up"] - parameter_bounds[param]["low"]
        )

    # Reverse log transform for L1, L2 (always positive)
    for param in ["L1", "L2"]:
        temp = np.log10(df[param])
        box_df[param + "_box"] = 0.5 + 0.5 * (temp - parameter_bounds[param]["exp_low"]) / (
            parameter_bounds[param]["exp_up"] - parameter_bounds[param]["exp_low"]
        )

    # Reverse signed log transform for L4, L7, L10, L11
    for param in ["L4", "L7", "L10", "L11"]:
        sign = np.sign(df[param])
        temp = np.log10(np.abs(df[param]))
        box_df[param + "_box"] = sign * (
            0.5 + 0.5 * (temp - parameter_bounds[param]["exp_low"]) / (
                parameter_bounds[param]["exp_up"] - parameter_bounds[param]["exp_low"]
            )
        )

    return box_df

def reverse_map_to_box_benchmarks(df, benchmark):
    box_df = pd.DataFrame(dtype=np.float64)

    # MH125 is fixed
    box_df["MH125_box"] = 1.0

    # MH10 still mapped linearly
    box_df["MH10_box"] = (df["MH10"] - parameter_bounds["MH10"]["low"]) / (
        parameter_bounds["MH10"]["up"] - parameter_bounds["MH10"]["low"]
    )

    if benchmark == "B":
        # MH20 = MH10 + 50 → box coordinate redundant
        box_df["MH20_box"] = 1.0  
        box_df["mC1_box"] = 1.0  
        box_df["mC2_box"] = 1.0  

    elif benchmark == "C":
        # MH20 = MH10 + 10, mC1 = MH10 + 50, mC2 = mC1 + 1
        box_df["MH20_box"] = 1.0
        box_df["mC1_box"] = 1.0
        box_df["mC2_box"] = 1.0

    elif benchmark == "G":
        # MH20 = MH10 + 2, mC1 = MH10 + 0.8, mC2 = mC1 + 0.5
        box_df["MH20_box"] = 1.0
        box_df["mC1_box"] = 1.0
        box_df["mC2_box"] = 1.0

    else:
        raise ValueError(f"Unknown benchmark {benchmark}")

    # Fixed theta
    box_df["theta_box"] = 1.0

    # g1, g2 mapping back
    box_df["g1_box"] = 0.5 + df["g1"] / 0.2 / 2
    box_df["g2_box"] = 0.5 + df["g2"] / 0.2 / 2

    # Fixed L1, L2, L4, L7, L10
    for p in ["L1", "L2", "L4", "L7", "L10"]:
        box_df[p + "_box"] = 1.0

    # L11 has signed log transform
    box_df["L11_box"] = 0.5 + np.sign(df["L11"]) * (np.abs(df["L11"]) / 0.1 / 2)

    return box_df



# from functools import reduce

# for scan in tqdm(scans):
#     print(scan)

#     # Find all good_points.csv files for this scan
#     all_good_points_files = glob.glob(os.path.join(f"data/{scan}/", "*", "good_points.csv"))
#     print(all_good_points_files)

#     # Get column sets for each file (nrows=1 makes it fast)
#     column_sets = [set(pd.read_csv(f, nrows=1).columns) for f in all_good_points_files]

#     # Compute intersection of columns across all files
#     common_columns = list(set.intersection(*column_sets))

#     # Load full data, but keep ONLY the intersection columns
#     all_good_points = pd.concat(
#         [pd.read_csv(f)[common_columns] for f in all_good_points_files],
#         ignore_index=True
#     )

#     print(f"→ {len(common_columns)} common columns kept.")

# dtype_map = {
#     "GoodHB": "double",
#     "chisq": "double",
#     "GoodPointNew": "double",
#     "generation": "double",
#     "episode_name": "string",
#     # add other problematic columns here
# }

for scan in tqdm(scans):
    print(scan)

    all_good_points_files = glob.glob(os.path.join(f"data/{scan}/", "*", "good_points.csv"))
    print(all_good_points_files)

    all_good_points = pd.concat([pd.read_csv(_file) for _file in all_good_points_files], ignore_index=True)
    # , dtype=dtype_map, low_memory=False
    # int_cols = all_good_points.select_dtypes(include=['int64']).columns
    # all_good_points[int_cols] = all_good_points[int_cols].astype('float64')

    if mode == "general":
        df_fixed = reverse_map_to_box_space(all_good_points)
        print(df_fixed)
    elif mode.startswith("benchmark:"):
        _, bench = mode.split(":")
        df_fixed = reverse_map_to_box_benchmarks(all_good_points, bench)
        print(df_fixed)
    else:
        raise ValueError(f"Unknown mode {mode}")

    df_fixed.to_csv(os.path.join(f"data/{scan}/", "good_points2.csv"))
    collect_files(all_good_points_files, os.path.join(f"data/{scan}/", "all_good_points.parquet"))


# # for scan in tqdm(scans):
# #     print(scan)
# #     all_good_points_files = glob.glob(os.path.join(f"data/{scan}/", "*", "good_points.csv"))

# #     print(all_good_points_files)

# #     all_good_points = pd.concat(
# #         [pd.read_csv(_file) for _file in all_good_points_files], ignore_index=True
# #     )

# #     df_fixed= reverse_map_to_box_space(all_good_points)
# #     print(df_fixed)

# #     df_fixed.to_csv(os.path.join(f"data/{scan}/", "good_points2.csv"))

# #     collect_files(all_good_points_files, os.path.join(f"data/{scan}/", "all_good_points.parquet"))

# #    all_good_points = pd.concat(
# #        [pd.read_csv(_file) for _file in all_good_points_files], ignore_index=True
# #    )
# #    save_parameters_fortran_file(
# #        all_good_points[parameter_columns], f"dados/{scan}.dat", True
# #    )
# #    all_good_points.to_parquet(os.path.join(f"data/{scan}/", "all_good_points.parquet"), index=False)

# # %%
