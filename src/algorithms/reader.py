import os
import time
import pandas as pd

from utils.process_points import evaluate_file
from utils.constraints import constraint_columns, constraint_HT_columns, constraint_MO_columns
from utils.data import save_parameters_fortran_file  # must preserve Fortran formatting


def reader(defaults):
    start_time = time.time()
    print("Got into reader")

    # === Input file and environment info ===
    # IN_PARAM_FILE = os.getenv("INPUT_FILE", "/app/configs/init.dat")
    IN_PARAM_FILE = "init.dat"
    if not os.path.exists(IN_PARAM_FILE):
        print(f"❌ Error: File {IN_PARAM_FILE} not found.")
        return

    experiment_name = os.getenv("experiment_name", defaults.get("experiment_name", "experiment"))
    episode_name = os.getenv("episode_name", "0")
    output_path = os.path.join("data", experiment_name, episode_name)
    os.makedirs(output_path, exist_ok=True)

    # === Constraint setup ===
    all_constraint_columns = constraint_columns.copy()
    if defaults.get("HT"):
        all_constraint_columns += constraint_HT_columns

    if defaults["MO"]:
        all_constraint_columns += constraint_MO_columns

    # tmp_file = os.path.join("configs", "init.dat")

    print(f"📂 Reading full dataset from: {IN_PARAM_FILE}")

    # === Read the full file into memory ===
    # df = pd.read_csv(IN_PARAM_FILE, sep=r"\s+", header=None, engine="python")
    # n_rows = len(df)
    # print(f"   Loaded {n_rows} rows.")

    # # === Save full dataset to Fortran-style input ===
    # save_parameters_fortran_file(df, tmp_file)
    # print(f"   Saved formatted Fortran file to: {tmp_file}")

    # === Run evaluator ===
    print("⚙️  Running evaluation...")
    results = evaluate_file(
        IN_PARAM_FILE, # tmp_file
        all_constraint_columns=all_constraint_columns,
        defaults=defaults,
    )

    # === Save results ===
    out_csv = os.path.join(output_path, "inspect_points_reader.csv")
    results.to_csv(out_csv, index=False)
    print(f"✅ Results written to: {out_csv}")

    # === Split good / bad points ===
    good = results.query("GoodPointNew == 1 and GoodPoint == 1")
    bad = results.query("GoodPointNew == 0 or GoodPoint == 0")

    if not good.empty:
        good_path = os.path.join(output_path, "good_points.csv")
        good.to_csv(good_path, index=False)
        print(f"   💚 Good points saved to: {good_path}")

    if not bad.empty:
        bad_path = os.path.join(output_path, "bad_points.csv")
        bad.to_csv(bad_path, index=False)
        print(f"   💔 Bad points saved to: {bad_path}")

    elapsed = time.time() - start_time
    print(f"\n🏁 Finished processing in {elapsed:.2f} seconds.")
