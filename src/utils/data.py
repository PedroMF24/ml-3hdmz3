import fortranformat as ff
import numpy as np
import pandas as pd

from .parameters import parameter_columns


def get_dataframe_from_fortran(file, column_names=None, nrows=None):

    # def fortran_float(s):
    #     try:
    #         return np.float64(s)
    #     except ValueError:
    #         return 0.0

    dtype_dict = {col: np.float64 for col in column_names if col not in ["id_H1_dom", "id_H2_dom", "id_dom_channel"]}
    dtype_dict.update({"id_H1_dom": str, "id_H2_dom": str, "id_dom_channel": str})


    df = pd.read_csv(
        file,
        header=None,
        #elim_whitespace=True,
        sep="\s+",
        #delim_whitespace=True,
        nrows=nrows,
        names=column_names,
        dtype=dtype_dict,
        # converters={f"absev({i})": fortran_float for i in range(1, 22)},
    ) # 
    return df



derived_parameters_columns = [
    "L3",
    "L5",
    "L6",
    "L8",
    "L9",
    "L12",
    "m11sq",
    "m22sq",
    "m33sq",
]


mu_columns = [
    "mu_ij(1,1)",
    "mu_ij(1,2)",
    "mu_ij(1,3)",
    "mu_ij(1,4)",
    "mu_ij(1,5)",
    "mu_ij(1,6)",
    "mu_ij(2,1)",
    "mu_ij(2,2)",
    "mu_ij(2,3)",
    "mu_ij(2,4)",
    "mu_ij(2,5)",
    "mu_ij(2,6)",
    "mu_ij(3,1)",
    "mu_ij(3,2)",
    "mu_ij(3,3)",
    "mu_ij(3,4)",
    "mu_ij(3,5)",
    "mu_ij(3,6)",
    "mu_ij(4,1)",
    "mu_ij(4,2)",
    "mu_ij(4,3)",
    "mu_ij(4,4)",
    "mu_ij(4,5)",
    "mu_ij(4,6)",
]


STUBr_columns = [
    "S",
    "T",
    "U",
    "BRXsgamma",
]


goodpoint_columns = [
    "GoodPoint",
    "GoodBFB",
    "GoodMin",
    "GoodUni",
    "GoodSTU",
    "GoodBSG",
    "GoodMus",
    "GoodDM",
]

unitarity_columns = [
    "absev(1)",
    "absev(2)",
    "absev(3)",
    "absev(4)",
    "absev(5)",
    "absev(6)",
    "absev(7)",
    "absev(8)",
    "absev(9)",
    "absev(10)",
    "absev(11)",
    "absev(12)",
    "absev(13)",
    "absev(14)",
    "absev(15)",
    "absev(16)",
    "absev(17)",
    "absev(18)",
    "absev(19)",
    "absev(20)",
    "absev(21)",
]

# unitarity_columns = [
#     "Ls(1)",
#     "Ls(2)",
#     "Ls(3)",
#     "Ls(4)",
#     "Ls(5)",
#     "Ls(6)",
#     "Ls(7)",
#     "Ls(8)",
#     "Ls(9)",
#     "Ls(10)",
#     "Ls(11)",
#     "Ls(12)",
#     "Ls(13)",
#     "Ls(14)",
#     "Ls(15)",
#     "Ls(16)",
#     "Ls(17)",
#     "Ls(18)",
#     "Ls(19)",
#     "Ls(20)",
#     "Ls(21)",
#     "Ls(22)",
#     "Ls(23)",
#     "Ls(24)",
#     "Ls(25)",
#     "Ls(26)",
#     "Ls(27)",
# ]

# kappa_columns = [
#     "kappaW",
#     "kappaU",
#     "kappaD",
#     "kappaL",
# ]

dm_columns = [
    "BR_hjinvisible(1)",
    "tauHplus(1)",
    "tauHplus(2)",
    "GammaTothj(1)",
    "GammaTotHplus(1)",
    "GammaTotHplus(2)"
]


all_columns = (
    parameter_columns
    + derived_parameters_columns
    + mu_columns
    + STUBr_columns
    + unitarity_columns
    + dm_columns
    + goodpoint_columns
    # + kappa_columns
)

observable_columns = mu_columns + STUBr_columns + unitarity_columns + dm_columns # + kappa_columns

neutralIds = [f"h{i+1}" for i in range(5)]
chargedIds = [f"Hp{i+1}" for i in range(2)]

scalarIds = neutralIds + chargedIds

HT_columns =  [f"selLim_{_id}_obsRatio" for _id in scalarIds] + ["chisqdiff"]

MO_columns = [
    "Omega_1",
    "Omega_2",
    "OmegaT",
    "dd_H1_SI_CS",
    "dd_H2_SI_CS",
]

MO_id_column = ["id_ann_CS"]

def save_parameters_fortran_file(parameters, filename, add_dummies=False):
    _parameters = parameters.copy()
    if add_dummies:
        _df_dummies = pd.DataFrame(
            data=np.zeros((len(_parameters), len(derived_parameters_columns))),
            columns=derived_parameters_columns,
            index=_parameters.index,
        )
        _parameters = _parameters.merge(_df_dummies, left_index=True, right_index=True)
    line_format = "(E17.8, "
    for _ in range(_parameters.shape[1] - 1):
        line_format += "E18.8, "
    line_format = line_format.strip(", ")
    line_format += ")"
    header_line = ff.FortranRecordWriter(line_format)
    Formatted_df = _parameters.apply(lambda x: header_line.write(x.values), axis=1)
    Formatted_df.to_csv(filename, index=False, header=False)
