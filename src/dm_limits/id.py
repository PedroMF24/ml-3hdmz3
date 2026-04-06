# ---------- Imports ----------
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.interpolate import interp1d
import os

# ---------- Global Storage ----------
dm_id_limit_functions = {}
# interpolated_curves = {}

# ---------- Unit Conversion ----------
def pb_to_cm3_per_s(pb): return pb * 1e-27
def tev_to_gev(tev): return tev * 1e3

# ---------- Apply Cutoff Masks ----------
def apply_mass_mask(name, masses, values):
    if "hess" in name:
        return masses[masses > 220.0], values[masses > 220.0]
    if "ww" in name:
        return masses[masses > 80.4], values[masses > 80.4]
    return masses, values


# ---------- Load and Interpolate Dataset ----------
def load_dataset(dataset, kind="cubic"):
    data = np.loadtxt(dataset["file"])
    masses, values = data[:, 0], data[:, 1]

    if "hess" in dataset["name"]:
        masses = tev_to_gev(masses)
    else:
        values = pb_to_cm3_per_s(values)

    masses, values = apply_mass_mask(dataset["name"], masses, values)

    log_interp = interp1d(np.log10(masses), np.log10(values), kind=kind,
                          bounds_error=False, fill_value="extrapolate")
    interp_fn = lambda m: 10 ** log_interp(np.log10(m))

    dm_id_limit_functions[dataset["name"]] = interp_fn

    mass_dense = np.logspace(np.log10(50), np.log10(1000), 1000)
    limit_dense = interp_fn(mass_dense)
    mass_dense, limit_dense = apply_mass_mask(dataset["name"], mass_dense, limit_dense)

    return mass_dense, limit_dense, dataset["label"], dataset["color"]

def prepare_all_datasets(datasets):
    return [load_dataset(ds) for ds in datasets]

# ---------- Plot Setup ----------
# def configure_plot(ax):
#     ax.set_yscale('log')
#     ax.set_xlim(50, 1000)
#     ax.set_ylim(1e-28, 1e-22)

#     xticks = list(range(50, 1001, 50))
#     labels = [str(x) if x in [50, 200, 400, 600, 800, 1000] else '' for x in xticks]
#     ax.set_xticks(xticks)
#     ax.xaxis.set_major_locator(FixedLocator(xticks))
#     ax.xaxis.set_major_formatter(FixedFormatter(labels))
#     ax.tick_params(axis='x', labelsize=10)

#     ax.set_xlabel(r"$m_{H_1}$ (GeV)")
#     ax.set_ylabel(r"$\left< \sigma v \right>_{\rm dom}$ [cm$^3$/s]")
#     ax.grid(True, which="both", linestyle="--", alpha=0.5)
#     ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

# def plot_all(ax, prepared_data):
#     for masses, values, label, color in prepared_data:
#         ax.plot(masses, values, '-', label=label, color=color, linewidth=2)
#     configure_plot(ax)

# ---------- Envelope Calculation ----------
def get_envelope(names, mass_grid):
    curves = [dm_id_limit_functions[name](mass_grid) for name in names if name in dm_id_limit_functions]
    return np.min(np.vstack(curves), axis=0) if curves else np.full_like(mass_grid, np.inf)

# def plot_envelopes(bb_names, ww_names, outpath):
#     mass_grid = np.linspace(50, 1000, 1000)
#     bb_env = get_envelope(bb_names, mass_grid)
    
#     ww_mask = mass_grid > 80.4
#     ww_env = get_envelope(ww_names, mass_grid)[ww_mask]
#     mass_ww = mass_grid[ww_mask]

#     fig, ax = plt.subplots(figsize=(9, 5))
#     ax.plot(mass_grid, bb_env, label="Combined $b\\bar{b}$ limit", color="black", linestyle="-")
#     ax.plot(mass_ww, ww_env, label="Combined $W^+W^-$ limit", color="red", linestyle="-")
#     configure_plot(ax)
#     plt.tight_layout()
#     plt.subplots_adjust(right=0.77)
#     plt.savefig(outpath)
#     plt.close()

# ---------- Point Check ----------
def check_limits(mass, sigma):
    print(f"Testing σ = {sigma:.1e} cm³/s at m = {mass:.0f} GeV")
    for name, fn in dm_id_limit_functions.items():
        limit = fn(mass)
        status = "ALLOWED" if sigma < limit else "EXCLUDED"
        print(f"  {name}: limit = {limit:.1e} → {status}")

# ---------- Main ----------
#def main():
datasets = [
    {"file": f"{os.path.dirname(__file__)}/indirect_detection_data/ams02_30_bb.txt", "label": "AMS-02 $b\\bar{b}$",   "name": "ams02_bb",   "color": "purple"},
    {"file": f"{os.path.dirname(__file__)}/indirect_detection_data/ams02_30_ww.txt", "label": "AMS-02 $W^+W^-$",      "name": "ams02_ww",   "color": "cyan"},
    {"file": f"{os.path.dirname(__file__)}/indirect_detection_data/fermilat_bb.txt", "label": "Fermi-LAT $b\\bar{b}$","name": "fermilat_bb","color": "black"},
    {"file": f"{os.path.dirname(__file__)}/indirect_detection_data/fermilat_ww.txt", "label": "Fermi-LAT $W^+W^-$",   "name": "fermilat_ww","color": "red"},
    {"file": f"{os.path.dirname(__file__)}/indirect_detection_data/limits_NFW_WW_IGS1420.txt", 
        "label": "H.E.S.S. $W^+W^-$", "name": "hess_ww", "color": "brown"},
]

data = prepare_all_datasets(datasets)

# Plot individual limits
# fig, ax = plt.subplots(figsize=(9, 5))
# plot_all(ax, data)
# plt.tight_layout()
# plt.subplots_adjust(right=0.77)
# plt.savefig("indirect_detection_data/combined_limits_loglin.png")
# plt.close()

# # Point test
# check_limits(500, 5e-10)

# Envelope
bb_names = ["ams02_bb", "fermilat_bb"]
ww_names = ["ams02_ww", "fermilat_ww", "hess_ww"]
mass_grid = np.linspace(50, 1000, 1000)

id_values_bb = get_envelope(bb_names, mass_grid)
log_interp_bb = interp1d(np.log10(mass_grid), np.log10(id_values_bb), kind="cubic",
bounds_error=False, fill_value="extrapolate")
interp_fn_bb = lambda m: 10 ** log_interp_bb(np.log10(m))
dm_id_limit_functions["bb"] = interp_fn_bb

id_values_ww = get_envelope(ww_names, mass_grid)
log_interp_ww = interp1d(np.log10(mass_grid), np.log10(id_values_ww), kind="cubic",
bounds_error=False, fill_value="extrapolate")
interp_fn_ww = lambda m: 10 ** log_interp_ww(np.log10(m))
dm_id_limit_functions["ww"] = interp_fn_ww
# plot_envelopes(bb_names, ww_names, "indirect_detection_data/combined_limits_envelope.png")

# if __name__ == "__main__":
#     main()
