import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.interpolate import interp1d
#from matplotlib.ticker import ScalarFormatter
import os

# This will store the interpolation functions
dm_dd_limit_functions = {}

def cm2_to_pb(value_cm2):
    return value_cm2 / 1e-36

def load_and_interpolate(filename, label, color, kind='cubic'):
    # Load data (handles both with or without header)
    data = np.loadtxt(filename)
    mass = data[:, 0]
    limit_cm2 = data[:, 1]
    limit_pb = cm2_to_pb(limit_cm2)

    # Interpolation in log-log space
    log_mass = np.log10(mass)
    log_limit = np.log10(limit_pb)
    interp_func = interp1d(log_mass, log_limit, kind=kind, bounds_error=False, fill_value='extrapolate')

    # Save callable to global dictionary
    dm_dd_limit_functions[label] = lambda m: 10**interp_func(np.log10(m))

    # Interpolated smooth curve
    mass_interp = np.logspace(np.log10(50), np.log10(1000), 1000)
    limit_interp = 10**interp_func(np.log10(mass_interp))

    return mass_interp, limit_interp, label, color

# Plot setup
# fig, ax = plt.subplots(figsize=(9, 5))

# Add all datasets here
datasets = [
    # (f"{os.path.dirname(__file__)}/direct_detection_data/LZ_2022.txt", "LZ 2022", "blue"),
    # (f"{os.path.dirname(__file__)}/direct_detection_data/PandaX.txt", "PandaX", "green"),
    # (f"{os.path.dirname(__file__)}/direct_detection_data/Xe_SI.txt", "Xe SI", "pink"),
    # (f"{os.path.dirname(__file__)}/direct_detection_data/XENON1T-paper.txt", "XENON1T", "purple"),
    # (f"{os.path.dirname(__file__)}/direct_detection_data/DARWIN.txt", "DARWIN", "red"),
    # (f"{os.path.dirname(__file__)}/direct_detection_data/neutrino-floor.txt", r"$\nu$ floor", "yellow"),
    (f"{os.path.dirname(__file__)}/direct_detection_data/LZ_2025.txt", "LZ 2025", "cyan")
]

for filename, label, color in datasets:
    mass_interp, limit_interp, label, color = load_and_interpolate(filename, label, color)
#     if color in ["red", "yellow"]:
#         # For DARWIN and neutrino floor, plot as dashed lines
#         ax.plot(mass_interp, limit_interp, '--', color=color, linewidth=2, label=label)
#     else:
#         ax.plot(mass_interp, limit_interp, '-', color=color, linewidth=2, label=label)

# # Axes and formatting
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(50, 1000)
# ax.set_ylim(1e-15, 1e-9)

# # x_ticks = list(range(50, 101, 10)) + list(range(200, 1001, 100))
# # ax.set_xticks(x_ticks)
# # ax.get_xaxis().set_major_formatter(ScalarFormatter())
# # ax.tick_params(axis='x', which='major', labelsize=10)

# # Custom ticks: keep ticks but label only selected ones
# x_ticks = list(range(50, 101, 10)) + list(range(200, 1001, 100))
# # [50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# x_tick_labels = []
# for x in x_ticks:
#     if x in [50, 100, 500, 1000]:
#         x_tick_labels.append(str(x))
#     else:
#         x_tick_labels.append('')


# ax.set_xticks(x_ticks)
# ax.xaxis.set_major_locator(FixedLocator(x_ticks))
# ax.xaxis.set_major_formatter(FixedFormatter(x_tick_labels))
# ax.tick_params(axis='x', which='major', labelsize=10)

# ax.set_xlabel(r"$m_{H_1}$ (GeV)")
# ax.set_ylabel(r"$\sigma_{\rm SI} \times f_H \times \xi$ (pb)")
# ax.grid(True, which="both", linestyle='--', alpha=0.5)
# ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

# plt.tight_layout()
# plt.subplots_adjust(right=0.83)  # leave space on the right for the legend
# plt.savefig("direct_detection_data/combined_limits_loglog.png")

# m_test = 500  # GeV
# sigma_test = 5e-10  # pb

# for exp_name, limit_fn in dm_dd_limit_functions.items():
#     limit = limit_fn(m_test)
#     status = "allowed" if sigma_test < limit else "excluded"
#     print(f"{exp_name}: σ = {sigma_test:.1e} pb at m = {m_test} GeV → {status}")
