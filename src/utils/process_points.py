import glob
import logging
import os
import shutil
import subprocess

import Higgs.bounds as HB
import numpy as np
import pandas as pd
from Higgs.tools.Input import predictionsFromDict, readHB5Datafiles
from sklearn.preprocessing import minmax_scale

from utils.constraints import (
    check_a1_b1_repulsion,
    check_all_constraints,
    check_HT,
    check_MO,
    #    check_HB,
    constraint_columns,
    constraints_bounds,
)
from utils.data import (
    all_columns,
    chargedIds,
    get_dataframe_from_fortran,
    neutralIds,
    save_parameters_fortran_file,
)
from utils.Fortran import path_fortran, run_HiggsTools
from utils.parameters import (
    get_box_dataframe,
    map_from_box_to_parameter_space,
    map_from_box_to_benchmarks,
    parameter_columns,
)

from interfaces.run_parse_mO_2DM import (
    mO_path,
    mO_tmp_path,
    run_micromegas, parse_micromegas_output,
    processMOml,write_mO_parameters_file
)

penalty_columns = ["penalty_parameter_density", "penalty_observable_density"]

experiment_name = os.environ.get("experiment_name", "0")
episode_name = os.environ.get("episode_name", "dev")


def evaluate_file(filename, all_constraint_columns, defaults):

    try:
        if not os.path.exists(f"{filename}"):
            print(f"Error: Initial parameter files {filename} not found")
            exit(1)

        _ = subprocess.run(f"./3HDM-Main {filename} out.dat".split(), capture_output=True)
        results = get_dataframe_from_fortran("out.dat", column_names=all_columns)

        if defaults["HT"]:
            # HT_results = do_HT5()
            HT_results = do_HT(defaults["Do_Chisq"])
            results = pd.concat([results, HT_results], axis=1)


        if defaults["MO"]:
            # run micromegas
            MO_results = do_MO(results, mO_tmp_path, 
                            dd_flag=defaults["DD"], id_flag=defaults["ID"], ch_relic_flag=defaults["Ch_Relic"])
            results = pd.concat([results, MO_results], axis=1)
            for f in glob.glob("Test_*"):
                os.remove(f)

        # if os.path.exists("out.dat"):
        #     os.remove("out.dat")
        
        check_all_constraints(results)

        if defaults["HT"]:
            check_HT(results)

        if defaults["MO"]:
            #            check_HB(results)
            check_MO(results, dd_flag=defaults["DD"], id_flag=defaults["ID"])

    except Exception:
        shutil.move("init.dat", f"data/{experiment_name}/{episode_name}/init.dat")
        logging.exception(
            f"Processing failed! Experiment: {experiment_name} | Episode: {episode_name}"
        )
        # logging.debug(results["selLim_h2_obsRatio"])
        # logging.debug(results["selLim_h2_obsRatio"].info())
        # print(results["selLim_h2_obsRatio"])
        exit(1)

    # Keep track of fitnesses before any transformation
    results["GoodPointNew"] = (results[all_constraint_columns] == 0).all(1).astype(int)
    results["ProportionValidConstraints"] = (
        results[constraint_columns].apply(lambda x: x == 0, axis=1).astype(int).mean(1)
    )
    results["MeanConstraints"] = results[all_constraint_columns].mean(axis=1)
    results["MaxConstraint"] = results[all_constraint_columns].max(axis=1)

    return results  


def evaluate_population_batch(
    population,
    all_constraint_columns,
    defaults,
    penalty_parameter_columns=None,
    penalty_observable_columns=None,
    penalty_parameter_density_estimator=None,
    penalty_observable_density_estimator=None,
):
    population_box_df = get_box_dataframe(population)
    # print("Pop box df")
    # print(population_box_df)

    if defaults["Benchmarks"]:
        population_df, map_ok = map_from_box_to_benchmarks(population_box_df, defaults["BH-Point"])
        if not map_ok:
            logging.exception(
                f"Processing BH map failed! Experiment: {experiment_name} | Episode: {episode_name}"
            )
            exit(1)
    else:
        population_df = map_from_box_to_parameter_space(population_box_df)

    # print("Pop df")
    # print(population_df)
 
    try:
        if os.path.exists("in.dat"):
            os.remove("in.dat")

        save_parameters_fortran_file(
            population_df[parameter_columns], "in.dat", add_dummies=True
        )
        
        print("Running process...")
        _ = subprocess.run("./3HDM-Main in.dat out.dat".split(), capture_output=True)
        results = get_dataframe_from_fortran("out.dat", column_names=all_columns)


        if defaults["HT"]:
            #            HT_results = do_HT5()
            print("Got into HT")
            HT_results = do_HT(defaults["Do_Chisq"])
            results = pd.concat([results, HT_results], axis=1)
            for f in glob.glob("Test_*"):
                os.remove(f)

        if defaults["MO"]:
            # run micromegas    
            MO_results = do_MO(results, mO_tmp_path, 
                            dd_flag=defaults["DD"],  id_flag=defaults["ID"], ch_relic_flag=defaults["Ch_Relic"])
            results = pd.concat([results, MO_results], axis=1)
            for f in glob.glob("Test_*"):
                os.remove(f)

        # print("Reached checkpoint in cmaes2, exit")
        # exit(1)

        # if os.path.exists("out.dat"):
        #     os.remove("out.dat")
        check_all_constraints(results)

        if defaults["HT"]:
            #            check_HB(results)
            check_HT(results)

        if defaults["MO"]:
            #            check_HB(results)
            check_MO(results, dd_flag=defaults["DD"], id_flag=defaults["ID"])


    except Exception:
        shutil.move("in.dat", f"data/{experiment_name}/{episode_name}/in.dat")
        logging.exception(
            f"Processing failed! Experiment: {experiment_name} | Episode: {episode_name}"
        )
        # logging.debug(results["selLim_h2_obsRatio"])
        # logging.debug(results["selLim_h2_obsRatio"].info())
        # print(results["selLim_h2_obsRatio"])
        exit(1)

    # Keep track of fitnesses before any transformation
    results["GoodPointNew"] = (results[all_constraint_columns] == 0).all(1).astype(int)
    results["ProportionValidConstraints"] = (
        results[constraint_columns].apply(lambda x: x == 0, axis=1).astype(int).mean(1)
    )
    results["MeanConstraints"] = results[all_constraint_columns].mean(axis=1)
    results["MaxConstraint"] = results[all_constraint_columns].max(axis=1)

    if penalty_parameter_density_estimator:
        raw_penalties = penalty_parameter_density_estimator.get_penalties(
            population_box_df[penalty_parameter_columns].values
        )
        results["penalty_parameter_density"] = raw_penalties
    else:
        results["penalty_parameter_density"] = 0.0

    if penalty_observable_density_estimator:
        raw_penalties = penalty_observable_density_estimator.get_penalties(
            results[penalty_observable_columns].values
        )
        results["penalty_observable_density"] = raw_penalties
    else:
        results["penalty_observable_density"] = 0.0

    return results


def evaluate_individuals(
    individuals,
    all_constraint_columns,
    defaults,
    penalty_parameter_columns=None,
    penalty_observable_columns=None,
    penalty_parameter_density_estimator=None,
    penalty_observable_density_estimator=None,
    scaler=None,
):
    results = evaluate_population_batch(
        population=individuals,
        penalty_parameter_columns=penalty_parameter_columns,
        penalty_observable_columns=penalty_observable_columns,
        penalty_parameter_density_estimator=penalty_parameter_density_estimator,
        penalty_observable_density_estimator=penalty_observable_density_estimator,
        all_constraint_columns=all_constraint_columns,
        defaults=defaults,
    )

    fitnesses = results[all_constraint_columns + penalty_columns].copy()

    # using minmax scaling to normalize the fitness values
    if scaler:
        fitnesses[all_constraint_columns] = scaler.transform(
            fitnesses[all_constraint_columns].values
        )

    else:
        fitnesses[all_constraint_columns] = minmax_scale(
            fitnesses[all_constraint_columns].values
        )
    fitnesses_constraints = list(
        fitnesses[all_constraint_columns].itertuples(index=False, name=None)
    )
    fitnesses_penalties = list(
        fitnesses[penalty_columns].itertuples(index=False, name=None)
    )

    for ind, fit_cons, fit_pen in zip(
        individuals, fitnesses_constraints, fitnesses_penalties
    ):

        fit_cons_sum = sum(fit_cons)
        fit_pen_mean = np.mean(fit_pen)
        fit = fit_cons_sum + fit_pen_mean
        # prevent penalty from dominating the fitness
        if fit_cons_sum != 0:
            fit += 1
        # ----------
        ind.fitness.values = (fit,)

    return individuals, results


def do_HT(Do_Chisq=False):

    resHT = run_HiggsTools(
        path=path_fortran,
        neutralCS=False,
        chargedCS=True,
        Couplings=True,
        neutralW=True,
        chargedW=True,
        flags=False,
        Do_Chisq=Do_Chisq,
    )

    result = pd.DataFrame(resHT)

    return result




def do_MO(df, file_path_template, dd_flag=False, id_flag=False, ch_relic_flag=False):
    all_results = []
    failed_points = []

    Omega_UB = 0.1236
    Omega_LB = 0.1164
    print("In do_MO")
    for point in df.index:
        print(f"Point: {df.index}")
        file_path = os.path.join(file_path_template, f"data_{point}.par")
        write_mO_parameters_file(df.iloc[point:point+1], file_path)
        # print selected columns
        cols_to_print = ["L1", "L2", "L4", "L7", "L10", "L11", 
                        "theta", "g1", "g2", "MH10", "MH20", "mC1", "mC2"]
        print(df.loc[point, cols_to_print])  # df.loc returns a Series for a single row
        
        output = run_micromegas(file_path, mO_path, dd_flag=dd_flag, id_flag=False)
        parsed_output = parse_micromegas_output(output, dd_flag=dd_flag, id_flag=False, ch_relic_flag=ch_relic_flag)

        OmegaT = parsed_output.get("relic_density", {}).get("OmegaT", None)

        # if OmegaT is None:
        #     print(f"Warning: OmegaT is None for point {point}. Skipping...")
            # failed_points.append({
            #     'point_index': point,
            #     'reason': 'OmegaT_is_None',
            #     'input_data': df.loc[point].to_dict(),
            #     'parsed_output': parsed_output
            # })
            # continue

        run_id = id_flag and (OmegaT is not None) and (Omega_LB <= OmegaT <= Omega_UB)
        timeout_flag = False
        print(f"OmegaT: {OmegaT}, run_id: {run_id}")
        if run_id:
            try:
                output_id = run_micromegas(file_path, mO_path, dd_flag=dd_flag, id_flag=True)
                parsed_output = parse_micromegas_output(output_id, dd_flag=dd_flag, id_flag=True, ch_relic_flag=ch_relic_flag)
            except subprocess.TimeoutExpired:
                print("micromegas ID run failed due to timeout.")
                timeout_flag = True

        print("In processing MOml")
        resMO = processMOml(parsed_output, dd_flag=dd_flag, id_flag=id_flag, ch_relic_flag=ch_relic_flag)
        resMO["run_id"] = run_id  # 👈 store whether this point passed relic
        resMO["timeout_flag"] = timeout_flag
        print("Passed processing MOml")
        all_results.append(resMO)


    result_df = pd.DataFrame(all_results, index=df.index)
    
    return result_df
