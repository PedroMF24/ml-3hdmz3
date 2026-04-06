import os

import numpy as np
import pandas as pd
import yaml
 
parameter_columns = [
    "MH125",
    "MH10",
    "MH20",
    "mC1",
    "mC2",
    "theta",
    "g1",
    "g2",
    "L1",
    "L2",
    "L4",
    "L7",
    "L10",
    "L11",
]
 

parameter_box_columns = [
    para + "_box" for para in parameter_columns if "MH125" not in para
]


parameter_bounds = yaml.safe_load(open("parameter-bounds.yml", "r"))
for p, bs in parameter_bounds.items():
    for bk, bv in bs.items():
        parameter_bounds[p][bk] = eval(bv) if isinstance(bv, str) else bv

if os.path.exists("parameter-bounds-local.yml"):
    parameter_bounds_local = yaml.safe_load(open("parameter-bounds-local.yml", "r"))
    if parameter_bounds_local:
        for p, bs in parameter_bounds_local.items():
            for bk, bv in bs.items():
                parameter_bounds[p][bk] = eval(bv) if isinstance(bv, str) else bv

defaults = yaml.safe_load(open("defaults.yml", "r"))
if os.path.exists("defaults-local.yml"):
    defaults_local = yaml.safe_load(open("defaults-local.yml", "r"))
    if defaults_local:
        for k, v in defaults_local.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    defaults[k][kk] = vv
            else:
                defaults[k] = v


def get_box_dataframe(population):
    _df = pd.DataFrame(data=population, columns=parameter_box_columns)
    _df["MH125_box"] = 1.0
    return _df[["MH125_box"] + parameter_box_columns]


def map_from_box_to_parameter_space(df):
    _df = pd.DataFrame(dtype=np.float64, columns=parameter_columns)

    _df["MH125"] = parameter_bounds["MH125"]["fixed"] * df["MH125_box"]
        
    mass_gap = 1.0  # minimal mass gap

    # -------------------
    # First generation: MH10
    _df["MH10"] = (
        parameter_bounds["MH10"]["low"]
        + (parameter_bounds["MH10"]["up"] - parameter_bounds["MH10"]["low"])
        * df["MH10_box"]
    )


    # -------------------
    # Second generation: MH20 >= MH10 + mass_gap
    _df["MH20"] = (
        parameter_bounds["MH20"]["low"]
        + (parameter_bounds["MH20"]["up"] - parameter_bounds["MH20"]["low"])
        * df["MH20_box"]
    )

    # Enforce MH20 >= MH10 + mass_gap
    _df["MH20"] = np.where(
        _df["MH20"] < _df["MH10"] + mass_gap,
        _df["MH10"] + mass_gap + (_df["MH20"] - parameter_bounds["MH20"]["low"]) * df["MH20_box"],
        _df["MH20"],
    )

    # -------------------
    # mC1 >= MH10 + mass_gap
    _df["mC1"] = (
        parameter_bounds["mC1"]["low"]
        + (parameter_bounds["mC1"]["up"] - parameter_bounds["mC1"]["low"])
        * df["mC1_box"]
    )

    _df["mC1"] = np.where(
        _df["mC1"] < _df["MH10"] + mass_gap,
        _df["MH10"] + mass_gap + (_df["mC1"] - parameter_bounds["mC1"]["low"]) * df["mC1_box"],
        _df["mC1"],
    )

    # -------------------
    # mC2 >= MH20 + mass_gap
    _df["mC2"] = (
        parameter_bounds["mC2"]["low"]
        + (parameter_bounds["mC2"]["up"] - parameter_bounds["mC2"]["low"])
        * df["mC2_box"]
    )

    _df["mC2"] = np.where(
        _df["mC2"] < _df["MH20"] + mass_gap,
        _df["MH20"] + mass_gap + (_df["mC2"] - parameter_bounds["mC2"]["low"]) * df["mC2_box"],
        _df["mC2"],
    )



    _df["theta"] = (
        parameter_bounds["theta"]["low"]
        + (parameter_bounds["theta"]["up"] - parameter_bounds["theta"]["low"])
        * df["theta_box"]
    )

    _df["g1"] = (
        parameter_bounds["g1"]["low"]
        + (parameter_bounds["g1"]["up"] - parameter_bounds["g1"]["low"])
        * df["g1_box"]
    )

    _df["g2"] = (
        parameter_bounds["g2"]["low"]
        + (parameter_bounds["g2"]["up"] - parameter_bounds["g2"]["low"])
        * df["g2_box"]
    )

    L1_temp = df["L1_box"] - 0.5
    # L1_temp_sign = np.sign(L1_temp)
    L1_temp = (
        parameter_bounds["L1"]["exp_low"]
        + (parameter_bounds["L1"]["exp_up"] - parameter_bounds["L1"]["exp_low"])
        * np.abs(L1_temp)
        * 2
    )
    _df["L1"] = 10**L1_temp

    L2_temp = df["L2_box"] - 0.5
    # L2_temp_sign = np.sign(L2_temp)
    L2_temp = (
        parameter_bounds["L2"]["exp_low"]
        + (parameter_bounds["L2"]["exp_up"] - parameter_bounds["L2"]["exp_low"])
        * np.abs(L2_temp)
        * 2
    )
    _df["L2"] = 10**L2_temp

    L4_temp = df["L4_box"] - 0.5
    L4_temp_sign = np.sign(L4_temp)
    L4_temp = (
        parameter_bounds["L4"]["exp_low"]
        + (parameter_bounds["L4"]["exp_up"] - parameter_bounds["L4"]["exp_low"])
        * np.abs(L4_temp)
        * 2
    )
    _df["L4"] = L4_temp_sign * 10**L4_temp

    L7_temp = df["L7_box"] - 0.5
    L7_temp_sign = np.sign(L7_temp)
    L7_temp = (
        parameter_bounds["L7"]["exp_low"]
        + (parameter_bounds["L7"]["exp_up"] - parameter_bounds["L7"]["exp_low"])
        * np.abs(L7_temp)
        * 2
    )
    _df["L7"] = L7_temp_sign * 10**L7_temp

    L10_temp = df["L10_box"] - 0.5
    L10_temp_sign = np.sign(L10_temp)
    L10_temp = (
        parameter_bounds["L10"]["exp_low"]
        + (parameter_bounds["L10"]["exp_up"] - parameter_bounds["L10"]["exp_low"])
        * np.abs(L10_temp)
        * 2
    )
    _df["L10"] = L10_temp_sign * 10**L10_temp

    L11_temp = df["L11_box"] - 0.5
    L11_temp_sign = np.sign(L11_temp)
    L11_temp = (
        parameter_bounds["L11"]["exp_low"]
        + (parameter_bounds["L11"]["exp_up"] - parameter_bounds["L11"]["exp_low"])
        * np.abs(L11_temp)
        * 2
    )
    _df["L11"] = L11_temp_sign * 10**L11_temp

    return _df

def map_from_box_to_parameter_space_(df):
    _df = pd.DataFrame(dtype=np.float64, columns=parameter_columns)

    _df["MH125"] = parameter_bounds["MH125"]["fixed"] * df["MH125_box"]
        
    mass_gap = 1.0  # minimal mass gap

    # -------------------
    # First generation: MH10
    _df["MH10"] = (
        parameter_bounds["MH10"]["low"]
        + (parameter_bounds["MH10"]["up"] - parameter_bounds["MH10"]["low"])
        * df["MH10_box"]
    )


    # -------------------
    # Second generation: MH20 >= MH10 + mass_gap
    _df["MH20"] = (
        parameter_bounds["MH20"]["low"]
        + (parameter_bounds["MH20"]["up"] - parameter_bounds["MH20"]["low"])
        * df["MH20_box"]
    )

    # Enforce MH20 >= MH10 + mass_gap
    _df["MH20"] = np.where(
        _df["MH20"] < _df["MH10"] + mass_gap,
        _df["MH10"] + mass_gap + (_df["MH20"] - parameter_bounds["MH20"]["low"]) * df["MH20_box"],
        _df["MH20"],
    )

    # -------------------
    # mC1 >= MH10 + mass_gap
    _df["mC1"] = (
        parameter_bounds["mC1"]["low"]
        + (parameter_bounds["mC1"]["up"] - parameter_bounds["mC1"]["low"])
        * df["mC1_box"]
    )

    _df["mC1"] = np.where(
        _df["mC1"] < _df["MH10"] + mass_gap,
        _df["MH10"] + mass_gap + (_df["mC1"] - parameter_bounds["mC1"]["low"]) * df["mC1_box"],
        _df["mC1"],
    )

    # -------------------
    # mC2 >= MH20 + mass_gap
    _df["mC2"] = (
        parameter_bounds["mC2"]["low"]
        + (parameter_bounds["mC2"]["up"] - parameter_bounds["mC2"]["low"])
        * df["mC2_box"]
    )

    _df["mC2"] = np.where(
        _df["mC2"] < _df["MH20"] + mass_gap,
        _df["MH20"] + mass_gap + (_df["mC2"] - parameter_bounds["mC2"]["low"]) * df["mC2_box"],
        _df["mC2"],
    )



    _df["theta"] = (
        parameter_bounds["theta"]["low"]
        + (parameter_bounds["theta"]["up"] - parameter_bounds["theta"]["low"])
        * df["theta_box"]
    )

    _df["g1"] = (
        parameter_bounds["g1"]["low"]
        + (parameter_bounds["g1"]["up"] - parameter_bounds["g1"]["low"])
        * df["g1_box"]
    )

    _df["g2"] = (
        parameter_bounds["g2"]["low"]
        + (parameter_bounds["g2"]["up"] - parameter_bounds["g2"]["low"])
        * df["g2_box"]
    )

    L1_temp = df["L1_box"] - 0.5
    # L1_temp_sign = np.sign(L1_temp)
    L1_temp = (
        parameter_bounds["L1"]["exp_low"]
        + (parameter_bounds["L1"]["exp_up"] - parameter_bounds["L1"]["exp_low"])
        * np.abs(L1_temp)
        * 2
    )
    _df["L1"] = 10**L1_temp

    L2_temp = df["L2_box"] - 0.5
    # L2_temp_sign = np.sign(L2_temp)
    L2_temp = (
        parameter_bounds["L2"]["exp_low"]
        + (parameter_bounds["L2"]["exp_up"] - parameter_bounds["L2"]["exp_low"])
        * np.abs(L2_temp)
        * 2
    )
    _df["L2"] = 10**L2_temp

    L4_temp = df["L4_box"] - 0.5
    L4_temp_sign = np.sign(L4_temp)
    L4_temp = (
        parameter_bounds["L4"]["exp_low"]
        + (parameter_bounds["L4"]["exp_up"] - parameter_bounds["L4"]["exp_low"])
        * np.abs(L4_temp)
        * 2
    )
    _df["L4"] = L4_temp_sign * 10**L4_temp

    L7_temp = df["L7_box"] - 0.5
    L7_temp_sign = np.sign(L7_temp)
    L7_temp = (
        parameter_bounds["L7"]["exp_low"]
        + (parameter_bounds["L7"]["exp_up"] - parameter_bounds["L7"]["exp_low"])
        * np.abs(L7_temp)
        * 2
    )
    _df["L7"] = L7_temp_sign * 10**L7_temp

    L10_temp = df["L10_box"] - 0.5
    L10_temp_sign = np.sign(L10_temp)
    L10_temp = (
        parameter_bounds["L10"]["exp_low"]
        + (parameter_bounds["L10"]["exp_up"] - parameter_bounds["L10"]["exp_low"])
        * np.abs(L10_temp)
        * 2
    )
    _df["L10"] = L10_temp_sign * 10**L10_temp

    L11_temp = df["L11_box"] - 0.5
    L11_temp_sign = np.sign(L11_temp)
    L11_temp = (
        parameter_bounds["L11"]["exp_low"]
        + (parameter_bounds["L11"]["exp_up"] - parameter_bounds["L11"]["exp_low"])
        * np.abs(L11_temp)
        * 2
    )
    _df["L11"] = L11_temp_sign * 10**L11_temp

    return _df



def map_from_box_to_benchmarks(df, benchmark):
    _df = pd.DataFrame(dtype=np.float64, columns=parameter_columns)

    _df["MH125"] = parameter_bounds["MH125"]["fixed"] * df["MH125_box"]
    
    # Points for Benchmark scenario B

    _df["MH10"] = (
        parameter_bounds["MH10"]["low"]
        + (parameter_bounds["MH10"]["up"] - parameter_bounds["MH10"]["low"])
        * df["MH10_box"]
    )

    if benchmark == "B":
        _df["MH20"] = (
            50.0 + _df["MH10"]
        )

        epsilon = 1e-3  # or any small offset

        _df["mC1"] = (
            60.0 + _df["MH10"]
        )


        _df["mC2"] = (
            10.0 + _df["mC1"]
        )
    elif benchmark == "C":
        _df["MH20"] = (
            10.0 + _df["MH10"]
        )

        epsilon = 1e-3  # or any small offset

        _df["mC1"] = (
            50.0 + _df["MH10"]
        )


        _df["mC2"] = (
            1.0 + _df["mC1"]
        )
    elif benchmark == "G":
        _df["MH20"] = (
            2.0 + _df["MH10"]
        )

        epsilon = 1e-3  # or any small offset

        _df["mC1"] = (
            0.8 + _df["MH10"]
        )


        _df["mC2"] = (
            0.5 + _df["mC1"]
        )
    else:
        return _df, False


    _df["theta"] = np.arccos(-1)/4.0
    

    g1_temp = df["g1_box"] - 0.5
    g1_temp_sign = np.sign(g1_temp)
    g1_temp = (
        (0.2) * np.abs(g1_temp)
        * 2
    )
    _df["g1"] = g1_temp_sign * g1_temp
    # _df["g1"] = 0.2 * df["g1_box"]
    # _df["g1"] = -(0.04 + (0.2 - 0.04) * df["g1_box"])


    g2_temp = df["g2_box"] - 0.5
    g2_temp_sign = np.sign(g2_temp)
    g2_temp = (
        (0.2) * np.abs(g2_temp)
        * 2
    )
    _df["g2"] = g2_temp_sign * g2_temp


    _df["L1"] = (
        0.13
    )

    _df["L2"] = (
        0.11
    )

    _df["L4"] = (
        0.12
    )

    _df["L7"] = (
        0.12
    )

    _df["L10"] = (
        0.1
    )

    L11_temp = df["L11_box"] - 0.5
    L11_temp_sign = np.sign(L11_temp)
    L11_temp = (
        (0.1) * np.abs(L11_temp)
        * 2
    )
    _df["L11"] = L11_temp_sign * L11_temp

    return _df, True
