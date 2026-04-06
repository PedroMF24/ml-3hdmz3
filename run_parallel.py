import datetime
import glob
import json
import os
import shutil
from multiprocessing import Pool, cpu_count

import docker
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# === Docker client & host info ===
client = docker.from_env()
host_working_dir = os.getcwd()
uid = os.getuid()
gid = os.getgid()

# === Load defaults ===
defaults = yaml.safe_load(open("configs/defaults-reader.yml", "r"))
if os.path.isfile("defaults-local.yml"):
    defaults_local = yaml.safe_load(open("defaults-local.yml", "r"))
    if defaults_local:
        for k, v in defaults_local.items():
            if isinstance(v, dict):
                defaults[k].update(v)
            else:
                defaults[k] = v

# === Experiment setup ===
ct = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
experiment_name = f"{ct}-{defaults['experiment_name']}"
output_path = os.path.join("reader", experiment_name)
os.makedirs(output_path, exist_ok=True)

# === Find .dat files ===
dat_folder = "check-points/theta" # Change folder of points
dat_files = sorted(glob.glob(os.path.join(dat_folder, "*.dat")))

n_files = len(dat_files)
n_cpus = min(n_files, defaults.get("n_cpus", cpu_count()))

print(f"Found {n_files} .dat files in {dat_folder}")
print(f"Running with {n_cpus} parallel workers.")

# === Helper to collect CSV -> Parquet ===
def collect_files(input_files_list, output_file):
    writer = None
    for input_file in input_files_list:
        df = pd.read_csv(input_file)
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema)
        writer.write_table(table)
    if writer:
        writer.close()

# === Build base image only once ===
image_name = "3hdmz3-ml-reader:dev"
dockerfile_path = "Dockerfile.reader"

print(f"Building base image {image_name} using {dockerfile_path}...")
client.images.build(path=".", tag=image_name, rm=True, dockerfile=dockerfile_path)
print("Base image built successfully!")

# === Temporary mount root ===
# tmp_mount_root = os.path.join(host_working_dir, "tmp_mounts")
# os.makedirs(tmp_mount_root, exist_ok=True)

# === Run container for each .dat file ===
def run_container(dat_path):
    basename = os.path.splitext(os.path.basename(dat_path))[0]
    container_name = f"{experiment_name}-{basename}"

    # Create an isolated tmp dir for this container
    # tmp_dir = os.path.join(tmp_mount_root, basename)
    # os.makedirs(tmp_dir, exist_ok=True)

    # Copy init.dat as required by the Fortran program
    # shutil.copy(dat_path, os.path.join(tmp_dir, "init.dat"))

    # Prepare volume mounts
    volumes = [
        f"{host_working_dir}/reader:/app/data",
        f"{host_working_dir}/aim:/app/aim",
        # f"{tmp_dir}:/app/configs",  # unique per container
        f"{os.path.join(host_working_dir, dat_path)}:/app/init.dat",  # input file
    ]

    # Optional local config mounts
    for fname in [
        "configs/defaults.yml",
        "configs/defaults-reader.yml",
        "configs/constraints-bounds.yml",
        "configs/parameter-bounds.yml",
        "configs/defaults-local.yml",
        "configs/constraints-bounds-local.yml",
        "configs/parameter-bounds-local.yml",
    ]:
        local_path = os.path.join(host_working_dir, fname)
        if os.path.exists(local_path):
            volumes.append(f"{local_path}:/app/{fname}")

    try:
        print(f"[{basename}] Running container {container_name}...")
        client.containers.run(
            image_name,
            remove=True,
            detach=False,
            name=container_name,
            user=f"{uid}:{gid}",
            environment={
                "experiment_name": experiment_name,
                "episode_name": basename,
                "SAMPLER": defaults.get("sampler", "reader"),
                # "INPUT_FILE": "/app/configs/init.dat",
            },
            volumes=volumes,
            nano_cpus=int(1e9),
        )
        print(f"[{basename}] Done!")
    except Exception as e:
        print(f"[{basename}] ERROR: {e}")
    # finally:
    #     # Clean up the temporary directory
    #     shutil.rmtree(tmp_dir, ignore_errors=True)

# === Run containers in parallel ===
with Pool(n_cpus) as p:
    p.map(run_container, dat_files)

# === Save defaults for record ===
json.dump(defaults, open(f"{output_path}/defaults.json", "w"))

print("✅ All containers completed successfully.")
