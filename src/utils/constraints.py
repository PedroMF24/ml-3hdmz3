import os

import numpy as np
import pandas as pd
import yaml

from .data import observable_columns, unitarity_columns
from dm_limits.dd import dm_dd_limit_functions
from dm_limits.id import dm_id_limit_functions 

# Used for comparison in C
NUMERICAL_INF = np.inf
# Used as infinite penalty
NUMERICAL_INF_LOG = np.log(np.finfo(np.float64).max) + 1
EPS = np.finfo(np.float64).eps

constraints_bounds = yaml.safe_load(open("constraints-bounds.yml", "r"))
for k, v in constraints_bounds.items():
    constraints_bounds[k] = eval(v) if isinstance(v, str) else v


if os.path.exists("constraints-bounds-local.yml"):
    constraints_bounds_local = yaml.safe_load(open("constraints-bounds-local.yml", "r"))
    if constraints_bounds_local:
        for k, v in constraints_bounds_local.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    constraints_bounds[k][kk] = eval(vv) if isinstance(vv, str) else vv
            else:
                constraints_bounds[k] = eval(v) if isinstance(v, str) else v


np.random.seed()


def C(O, O_LB, O_UB):
    #    print(O, O_LB, O_UB)
    vC = np.vectorize(lambda x: max(0.0, -x + O_LB, x - O_UB), otypes=[np.float64])
    return np.log(1 + vC(O))


vmin = np.vectorize(min, otypes=[np.float64])

vmax = np.vectorize(max, otypes=[np.float64])


def check_kappas(df):
    kappa_W_centre = constraints_bounds["kappa_W_centre"]
    error_W_upper = constraints_bounds["error_W_upper"]
    error_W_lower = constraints_bounds["error_W_lower"]

    kappa_Z_centre = constraints_bounds["kappa_Z_centre"]
    error_Z_upper = constraints_bounds["error_Z_upper"]
    error_Z_lower = constraints_bounds["error_Z_lower"]

    kappa_b_centre = constraints_bounds["kappa_b_centre"]
    error_b_upper = constraints_bounds["error_b_upper"]
    error_b_lower = constraints_bounds["error_b_lower"]

    kappa_t_centre = constraints_bounds["kappa_t_centre"]
    error_t_upper = constraints_bounds["error_t_upper"]
    error_t_lower = constraints_bounds["error_t_lower"]

    kappa_tau_centre = constraints_bounds["kappa_tau_centre"]
    error_tau_upper = constraints_bounds["error_tau_upper"]
    error_tau_lower = constraints_bounds["error_tau_lower"]

    kappa_W_min = kappa_W_centre - constraints_bounds["kappa_i_sigma"] * error_W_lower
    kappa_W_max = kappa_W_centre + constraints_bounds["kappa_i_sigma"] * error_W_upper

    kappa_Z_min = kappa_Z_centre - constraints_bounds["kappa_i_sigma"] * error_Z_lower
    kappa_Z_max = kappa_Z_centre + constraints_bounds["kappa_i_sigma"] * error_Z_upper

    kappa_b_min = kappa_b_centre - constraints_bounds["kappa_i_sigma"] * error_b_lower
    kappa_b_max = kappa_b_centre + constraints_bounds["kappa_i_sigma"] * error_b_upper

    kappa_b_min_WS = (
        -kappa_b_centre - constraints_bounds["kappa_i_sigma"] * error_b_lower
    )
    kappa_b_max_WS = (
        -kappa_b_centre + constraints_bounds["kappa_i_sigma"] * error_b_upper
    )

    kappa_t_min = kappa_t_centre - constraints_bounds["kappa_i_sigma"] * error_t_lower
    kappa_t_max = kappa_t_centre + constraints_bounds["kappa_i_sigma"] * error_t_upper

    kappa_tau_min = (
        kappa_tau_centre - constraints_bounds["kappa_i_sigma"] * error_tau_lower
    )
    kappa_tau_max = (
        kappa_tau_centre + constraints_bounds["kappa_i_sigma"] * error_tau_upper
    )

    CkappaW = C(df["kappaW"], kappa_W_min, kappa_W_max)

    CkappaZ = C(df["kappaW"], kappa_Z_min, kappa_Z_max)

    CkappaU = C(df["kappaU"], kappa_t_min, kappa_t_max)

    CkappaDNS = C(df["kappaD"], kappa_b_min, kappa_b_max)
    CkappaDWS = C(df["kappaD"], kappa_b_min_WS, kappa_b_max_WS)
    CkappaD = vmin(CkappaDNS, CkappaDWS)

    CkappaL = C(df["kappaL"], kappa_tau_min, kappa_tau_max)

    return CkappaW, CkappaZ, CkappaU, CkappaD, CkappaL


def check_bound_from_below(df):
    def check_positivity(x11, x22, x33, x12, x13, x23):
        mask1 = (x11 > 0.0) & (x22 > 0.0) & (x33 > 0.0)
        x12bar = np.where(mask1, np.sqrt(x11 * x22) + x12, np.nan)
        x13bar = np.where(mask1, np.sqrt(x11 * x33) + x13, np.nan)
        x23bar = np.where(mask1, np.sqrt(x22 * x33) + x23, np.nan)
        mask2 = (x12bar >= 0.0) & (x13bar >= 0.0) & (x23bar >= 0.0)
        aux = np.where(
            mask1 & mask2,
            np.sqrt(x11 * x22 * x33)
            + x12 * np.sqrt(x33)
            + x13 * np.sqrt(x22)
            + x23 * np.sqrt(x11)
            + np.sqrt(2.0 * x12bar * x13bar * x23bar),
            np.nan,
        )

        cx11 = C(x11, 0.0, NUMERICAL_INF)
        cx22 = C(x22, 0.0, NUMERICAL_INF)
        cx33 = C(x33, 0.0, NUMERICAL_INF)
        cx12bar = np.where(mask1, C(x12bar, 0.0, NUMERICAL_INF), NUMERICAL_INF_LOG)
        cx13bar = np.where(mask1, C(x13bar, 0.0, NUMERICAL_INF), NUMERICAL_INF_LOG)
        cx23bar = np.where(mask1, C(x23bar, 0.0, NUMERICAL_INF), NUMERICAL_INF_LOG)
        cxaux = np.where(mask1 & mask2, C(aux, 0.0, NUMERICAL_INF), NUMERICAL_INF_LOG)
        return cx11, cx22, cx33, cx12bar, cx13bar, cx23bar, cxaux

    A11 = 2 * df["L1"]
    A22 = 2 * df["L2"]
    A33 = 2 * df["L3"]
    A12 = df["L4"] + df["L7"] + np.where(-df["L7"] >= 0, 0, -df["L7"]) - 2 * df["L10"].apply(np.abs) - 2 * df["L11"].apply(np.abs)
    A13 = df["L5"] + df["L8"] + np.where(-df["L8"] >= 0, 0, -df["L8"]) - 2 * df["L10"].apply(np.abs) - 2 * df["L12"].apply(np.abs)
    A23 = df["L6"] + df["L9"] + np.where(-df["L9"] >= 0, 0, -df["L9"]) - 2 * df["L11"].apply(np.abs) - 2 * df["L12"].apply(np.abs)


    cA11, cA22, cA33, cA12bar, cA13bar, cA23bar, cAaux = check_positivity(
        A11, A22, A33, A12, A13, A23
    )

    Lpp12 = -df["L7"]
    Lpp13 = -df["L8"]
    Lpp23 = -df["L9"]

    B11 = A11
    B22 = A22
    B33 = A33
    B12 = A12 + vmin(0, Lpp12) - 2 * df["L10"] ** 2
    B13 = A13 + vmin(0, Lpp13) - 2 * df["L11"] ** 2
    B23 = A23 + vmin(0, Lpp23) - 2 * df["L12"] ** 2

    cB11, cB22, cB33, cB12bar, cB13bar, cB23bar, cBaux = check_positivity(
        B11, B22, B33, B12, B13, B23
    )

    return (
        cA11,
        cA22,
        cA33,
        cA12bar,
        cA13bar,
        cA23bar,
        cAaux,
        # cB11,
        # cB22,
        # cB33,
        # cB12bar,
        # cB13bar,
        # cB23bar,
        # cBaux,
    )


def check_global_min(df):  # L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, m11sq, m22sq, m33sq

    L1 = df["L1"]
    L2 = df["L2"]
    L3_first = df["L3"].iloc[0]

    m11sq = df["m11sq"]
    m22sq = df["m22sq"]
    m33sq_first = df["m33sq"].iloc[0]

    VminN = -m33sq_first**2 / (4.0 * L3_first)
    
    r1 = -m11sq / (2.0 * L1)
    VTest1 = np.where(r1 > 0.0, -m11sq**2 / (4.0 * L1), 0.0)
    r2 = -m22sq / (2.0 * L2)
    VTest2 = np.where(r2 > 0.0, -m22sq**2 / (4.0 * L2), 0.0)

    Vmin_ratio_LB = 1.0 # constraints_bounds["Vmin_ratio_lower"]
    Vmin_ratio_UB = NUMERICAL_INF # constraints_bounds["Vmin_ratio_upper"]
    bounds = [C(VTest / VminN, -NUMERICAL_INF, 1.0) for VTest in np.vstack([VTest1, VTest2]).min(axis=0)]
    # bounds = [C(vtest / VminN, Vmin_ratio_LB, Vmin_ratio_UB) for vtest in np.vstack([VTest1, VTest2]).min(axis=0)]

    # print(bounds)

    return tuple(np.array(bounds).reshape(1,-1))


def check_dm(df):
    mW=80.35797
    mZ=91.15348

    Cmassdiff_1 = C(df["MH10"] + df["mC1"] - mW, 0.0, NUMERICAL_INF)
    Cmassdiff_2 = C(df["MH20"] + df["mC2"] - mW, 0.0, NUMERICAL_INF)

    Cmassdiff_3 = C(df["MH10"] + df["mC1"] - mW, 0.0, NUMERICAL_INF) # mA1 = mH1
    Cmassdiff_4 = C(df["MH20"] + df["mC2"] - mW, 0.0, NUMERICAL_INF) # mA2 = mH2

    Cmassdiff_5 = C(2*df["MH10"] - mZ, 0.0, NUMERICAL_INF)
    Cmassdiff_6 = C(2*df["MH20"] - mZ, 0.0, NUMERICAL_INF)

    Cmassdiff_7 = C(2*df["mC1"] - mZ, 0.0, NUMERICAL_INF)
    Cmassdiff_8 = C(2*df["mC2"] - mZ, 0.0, NUMERICAL_INF)

    # Last constraint e.g. eq. 55 of 2407.15933 is never met as mAi = mHi

    CBR_h_inv = C(df["BR_hjinvisible(1)"], 0.0, 0.184)
    # CtauHP1 = C(df["tauHplus(1)"], 0.0, 1e-7)
    # CtauHP2 = C(df["tauHplus(2)"], 0.0, 1e-7)
    CGamma_h = C(df["GammaTothj(1)"], 0.0, 0.0075)

    CtauHP1 = C(6.582e-25/df["GammaTotHplus(1)"], 0.0, 1e-7)
    CtauHP2 = C(6.582e-25/df["GammaTotHplus(2)"], 0.0, 1e-7) 


    # print("Start here")
    # print(df["tauHplus(1)"])
    # print(df["tauHplus(2)"])
    # print(df["GammaTotHplus(1)"])
    # print(df["GammaTotHplus(2)"])
    # print(6.582e-25/df["GammaTotHplus(1)"])
    # print(6.582e-25/df["GammaTotHplus(2)"])
    # print("End here")

    Ccheck_dm_tuple = (
        Cmassdiff_1,
        Cmassdiff_2,
        Cmassdiff_3,
        Cmassdiff_4,
        Cmassdiff_5,
        Cmassdiff_6,
        Cmassdiff_7,
        Cmassdiff_8,
        CBR_h_inv,
        # CtauHP1,
        CtauHP1,
        # CtauHP2,
        CtauHP2,
        CGamma_h,
    )

    return Ccheck_dm_tuple



def check_oblique_parameters(df):
    U = constraints_bounds["U_centre"]
    ULB = U - constraints_bounds["U_lower"]
    UUB = U + constraints_bounds["U_upper"]

    CU = C(df["U"], ULB, UUB)

    a1 = constraints_bounds["a1"]
    a2 = constraints_bounds["a2"]
    a3 = constraints_bounds["a3"]
    a4 = constraints_bounds["a4"]
    a5 = constraints_bounds["a5"]
    a6 = constraints_bounds["a6"]

    Corr = (
        a1 * df["S"] ** 2
        + a2 * df["S"] * df["T"]
        + a3 * df["T"] ** 2
        + a4 * df["S"]
        + a5 * df["T"]
        + a6
    )

    CCorr = C(Corr, 0.0, NUMERICAL_INF)

    return CU, CCorr


def check_unitariy(df):
    cevs = [C(df[e].abs(), 0.0, 8 * np.pi) for e in unitarity_columns]
    return tuple(cevs)


def check_signal_strenghts(df):
    Cs = [
        C(
            df[k],
            constraints_bounds[k + "_centre"]
            - constraints_bounds["mu_i_sigma"] * constraints_bounds[k + "_lower"],
            constraints_bounds[k + "_centre"]
            + constraints_bounds["mu_i_sigma"] * constraints_bounds[k + "_upper"],
        )
        for k in observable_columns
        if "mu_ij" in k
    ]
    return tuple(Cs)


def check_bsg(df):
    BRbsgUB = constraints_bounds["BRbsgUB"] / (
        1.0 - constraints_bounds["BRbsg_epsilon"]
    )
    BRbsgLb = constraints_bounds["BRbsgLb"] / (
        1.0 + constraints_bounds["BRbsg_epsilon"]
    )
    return C(df["BRXsgamma"], BRbsgLb, BRbsgUB)

def check_frontier_up(df):
    # frontier =  1.416666667 * df["Omega_1"] + 0.02583333333
    # frontier = 3.666666667 * df["Omega_1"] + 0.003333333333
    
    # frontier = -0.038 * df["MH10"] + 2.442
    # frontier = 5.13513514e-4 * df["MH10"] -0.0927027027
    m=(0.6 - 0.58)/(700-450)
    b = 0.6 - m*700
    frontier = m * df["MH10"] + b
    val = df["theta"]/frontier

    # 4.5e-4 * df["MH10"] - 0.075
    # frontier = 6.48148148e-4 * df["MH10"] - 0.1046296296 # 4.5e-4 * df["MH10"] - 0.075
    # -0.009 * df["MH10"] + 0.55
    # val = df["OmegaT"]/frontier
    # val = df["OmegaT"]/frontier
    # return C(val, 1, np.inf) 
    return C(val, 0, 1)  

def check_frontier_down(df):
    # frontier =  1.416666667 * df["Omega_1"] + 0.02583333333
    # frontier = 3.666666667 * df["Omega_1"] + 0.003333333333
    
    # frontier = -0.038 * df["MH10"] + 2.442
    # frontier = 5.13513514e-4 * df["MH10"] -0.0927027027
    m=(0.56 - 0.54)/(585-540)
    b = 0.56 - m*585
    frontier = m * df["MH10"] + b
    val = df["theta"]/frontier

    # 4.5e-4 * df["MH10"] - 0.075
    # frontier = 6.48148148e-4 * df["MH10"] - 0.1046296296 # 4.5e-4 * df["MH10"] - 0.075
    # -0.009 * df["MH10"] + 0.55
    # val = df["OmegaT"]/frontier
    # val = df["OmegaT"]/frontier
    return C(val, 1, np.inf) 
 

@np.errstate(all="ignore")
def check_all_constraints(df):
    # (
    #     df["CMH30"],
    #     df["CMH40"],
    #     df["CMH50"],
    # ) = check_negative_masses(df)

    # (
    #     df["CkappaW"],
    #     df["CkappaZ"],
    #     df["CkappaU"],
    #     df["CkappaD"],
    #     df["CkappaL"],
    # ) = check_kappas(df)

    (
        df["CA11"],
        df["CA22"],
        df["CA33"],
        df["CA12bar"],
        df["CA13bar"],
        df["CA23bar"],
        df["CAaux"],
        # df["CB11"],
        # df["CB22"],
        # df["CB33"],
        # df["CB12bar"],
        # df["CB13bar"],
        # df["CB23bar"],
        # df["CBaux"],
    ) = check_bound_from_below(df)

    (
        # df["Cgmin_1"],
        # df["Cgmin_2"],
        df["Cgmin"],
    ) = check_global_min(df)


    # (
    #     df["COmega"],
    #     df["Cdd_SI_CS"],
    #     #df["Cid_ann_CS"],
    # ) = check_MO(df)


    df["CU"], df["CCorr"] = check_oblique_parameters(df)

    df[[f"C{col}" for col in unitarity_columns]] = np.array(check_unitariy(df)).T

    df[[f"C{col}" for col in observable_columns if "mu_ij" in col]] = np.array(
        check_signal_strenghts(df)
    ).T

    df["CBRXsgamma"] = check_bsg(df)

    (
        df["Cmassdiff_1"],
        df["Cmassdiff_2"],
        df["Cmassdiff_3"],
        df["Cmassdiff_4"],
        df["Cmassdiff_5"],
        df["Cmassdiff_6"],
        df["Cmassdiff_7"],
        df["Cmassdiff_8"],
        df["CBR_hjinvisible(1)"],
        # df["tauHplus(1)"],
        df["CtauHplus(1)"],
        # df["tauHplus(2)"],
        df["CtauHplus(2)"],
        df["CGammaTothj(1)"],
    ) = check_dm(df)

    # df["CFrontier_up"] = check_frontier_up(df)
    # df["CFrontier_down"] = check_frontier_down(df)
    # df["CEDM"] = check_EDM(df)
    
    # df["mass_diff"] = df[['MH20', 'MH30', 'MH40', 'MH50']].min(axis=1)-125
    # df["Cmass_diff"]= check_mass_difference(df)


    # df["Cghbb_s"]= check_wrongsign(df)
    # df["Cghbb_p"]= check_pseudoscalar_b(df)
    
    # df["Cghee_s"]= check_wrongsign_tau(df)
    # df["Cghee_p"]= check_pseudoscalar_tau(df)
    # df["ghbb_circle"] = np.sqrt(df["ghjbb_s(1)"]*df["ghjbb_s(1)"]+df["ghjbb_p(1)"]*df["ghjbb_p(1)"])
    # df["Cghbb_circle"]= check_circle(df)


def check_a1_b1_repulsion(df):
    C_b1_a1_lower = C(
        df["alp1"] / (df["bet1"] + EPS),
        -NUMERICAL_INF,
        constraints_bounds["b1_a1_centre"] - constraints_bounds["b1_a1_lower"],
    )


def check_HT(df):
    #    print(df["selLim_h1_obsRatio"])
    df["CselLim_h1_obsRatio"] = C(df["selLim_h1_obsRatio"], -NUMERICAL_INF, 1)
    df["CselLim_h2_obsRatio"] = C(df["selLim_h2_obsRatio"], -NUMERICAL_INF, 1)
    df["CselLim_h3_obsRatio"] = C(df["selLim_h3_obsRatio"], -NUMERICAL_INF, 1)
    df["CselLim_h4_obsRatio"] = C(df["selLim_h4_obsRatio"], -NUMERICAL_INF, 1)
    df["CselLim_h5_obsRatio"] = C(df["selLim_h5_obsRatio"], -NUMERICAL_INF, 1)
    df["CselLim_Hp1_obsRatio"] = C(df["selLim_Hp1_obsRatio"], -NUMERICAL_INF, 1)
    df["CselLim_Hp2_obsRatio"] = C(df["selLim_Hp2_obsRatio"], -NUMERICAL_INF, 1)
    df["Cchisqdiff"] = C(df["chisqdiff"], 0, constraints_bounds["chisq_ub"])



def check_MO(df, dd_exp="LZ 2025", dd_flag=False, id_flag=False):
    #dark matter mass

    # Omega
    OmegaLB = constraints_bounds["Omega_lower"]
    OmegaUB = constraints_bounds["Omega_upper"]
    Omega = df["Omega_1"] + df["Omega_2"]
    # COmega = 
    print("Before COmega calculation")
    df["COmega"] = np.where(
            df["OmegaT"].apply(pd.isna), NUMERICAL_INF_LOG, C(Omega, OmegaLB, OmegaUB)
        )

    # df["COmega"] = COmega

    # --- Direct detection ---
    if dd_flag:
        def check_H1_dd_cs(subdf):
            # dd_lim = dm_dd_limit_functions[dd_exp](subdf["MH10"])
            # dd_H1 = subdf["dd_H1_SI_CS"] / dd_lim
            dd_lim = dm_dd_limit_functions[dd_exp](subdf["MH10"])

            # dd_A1_lim = dm_dd_limit_functions[dd_exp](df["MH10"])
            dd_H1 = df["dd_H1_SI_CS"]/dd_lim

            # dd_A1 = df["dd_A1_SI_CS"]/dd_A1_lim
            ddLB = constraints_bounds["dd_SI_CS_lower"]
            ddUB = constraints_bounds["dd_SI_CS_upper"]
            return C(dd_H1, ddLB, ddUB)
        
        def check_A1_dd_cs(subdf):
            # dd_lim = dm_dd_limit_functions[dd_exp](subdf["MH10"])
            # dd_H1 = subdf["dd_H1_SI_CS"] / dd_lim
            dd_lim = dm_dd_limit_functions[dd_exp](subdf["MH10"])

            # dd_A1_lim = dm_dd_limit_functions[dd_exp](df["MH10"])
            dd_A1 = df["dd_H2_SI_CS"]/dd_lim

            # dd_A1 = df["dd_A1_SI_CS"]/dd_A1_lim
            ddLB = constraints_bounds["dd_SI_CS_lower"]
            ddUB = constraints_bounds["dd_SI_CS_upper"]
            return C(dd_A1, ddLB, ddUB) 

        df["Cdd_H1_SI_CS"] = np.where(
            df["dd_H1_SI_CS"].apply(pd.isna), NUMERICAL_INF_LOG, check_H1_dd_cs(df)
        )
        df["Cdd_H2_SI_CS"] = np.where(
            df["dd_H2_SI_CS"].apply(pd.isna), NUMERICAL_INF_LOG, check_A1_dd_cs(df)
        )
        # df["Cdd_H2_SI_CS"] = 0.0
    else:
        df["Cdd_H1_SI_CS"] = 0.0
        df["Cdd_H2_SI_CS"] = 0.0

   # --- Indirect detection ---
   # and "run_id" in df.columns
    df["Cid_ann_CS"] = 0.0
    if id_flag:
        df = df.fillna(np.nan)
        mask = df["run_id"].astype(bool)

        if mask.any():
            def check_id_cs(subdf):
                allowed_channels = {
                    "b b~": dm_id_limit_functions["bb"],
                    "WP WP~": dm_id_limit_functions["ww"],
                    "Z Z": dm_id_limit_functions["ww"],
                }
                funcs = subdf["id_H1_dom"].map(allowed_channels)
                id_H1_lim = [
                    f(m) if callable(f) else NUMERICAL_INF_LOG
                    for f, m in zip(funcs, subdf["MH10"])
                ]
                id_H1 = subdf["id_ann_CS"] / id_H1_lim
                idLB = constraints_bounds["id_ann_CS_lower"]
                idUB = constraints_bounds["id_ann_CS_upper"]
                return C(id_H1, idLB, idUB)

            df.loc[mask, "Cid_ann_CS"] = np.where(
            df.loc[mask, "id_ann_CS"].apply(np.isnan), NUMERICAL_INF_LOG, check_id_cs(df.loc[mask]))
            
    else:
        df["Cid_ann_CS"] = 0.0

    return df["COmega"], df["Cdd_H1_SI_CS"], df["Cdd_H2_SI_CS"], df["Cid_ann_CS"]


# check_HT
# Name: selLim_h1_obsRatio, dtype: object -inf 1

# mask_id = df["id_H1_dom"].apply(lambda x: np.isnan(x) if type(x) != str else False)
# mask_id = [True if (("b" not in _) and ("W" not in _)) else False for _ in df["id_H1_dom"].fillna("-")]
# Cid_ann_CS = np.where(mask_id, NUMERICAL_INF_LOG, check_id_cs(df))


constraint_columns = (
    [
        # "CkappaW",  # kappa
        # "CkappaZ",
        # "CkappaU",
        # "CkappaD",
        # "CkappaL",
        "CA11",  # BFB
        "CA22",
        "CA33",
        "CA12bar",
        "CA13bar",
        "CA23bar",
        "CAaux",
        # "CB11",
        # "CB22",
        # "CB33",
        # "CB12bar",
        # "CB13bar",
        # "CB23bar",
        # "CBaux",
        "CU",  # STU
        "CCorr",
        "Cgmin",  # global min
    ]
    + 
    [
        "Cmassdiff_1",  # DM
        "Cmassdiff_2",
        "Cmassdiff_3",
        "Cmassdiff_4",
        "Cmassdiff_5",
        "Cmassdiff_6",
        "Cmassdiff_7",
        "Cmassdiff_8",
        "CBR_hjinvisible(1)",
        # "CtauHplus(1)",
        "CtauHplus(1)",
        # "CtauHplus(2)",
        "CtauHplus(2)",
        "CGammaTothj(1)",
    ]

    + [f"C{col}" for col in unitarity_columns]  # UNIT coefs paper miguel
    + [f"C{col}" for col in observable_columns if "mu_ij" in col]  # Atlas muij
    + ["CBRXsgamma"]  # BSgamma

    # + ["CFrontier_up"]
    # + ["CFrontier_down"]
    

)

constraint_HT_columns = [
    "CselLim_h1_obsRatio",
    "CselLim_h2_obsRatio",
    "CselLim_h3_obsRatio",
    "CselLim_h4_obsRatio",
    "CselLim_h5_obsRatio",
    "CselLim_Hp1_obsRatio",
    "CselLim_Hp2_obsRatio",
    "Cchisqdiff",
]

constraint_MO_columns = [
    "COmega",
    "Cdd_H1_SI_CS",
    "Cdd_H2_SI_CS",
    "Cid_ann_CS",
]

# constraint_MO_id_column = ["Cid_ann_CS"]


constraint_a1_b1_repulsion = ["C_a1_b1"]
