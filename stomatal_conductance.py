"""Routines for testing performance of stomatal conductance models on data.
"""

from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def remove_outliers(df0: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Removes rows where any column has data greater than 3 standard deviations
    from the mean."""
    dfr = df0[((np.abs(df0 - df0.mean()) / df0.std()) < 3).all(axis=1)]
    nrm = len(df0) - len(dfr)
    if verbose:
        print(f"Removed {nrm} rows ({100*nrm/len(df0):.1f}%) marked as outliers.")
    return dfr


def add_relative_humidity(
    df0: pd.DataFrame, temperature_col: str, vpd_col: str, rh_col: str
) -> pd.DataFrame:
    """Add relative humidity [%] to the dataframe using an equation based on
    temperature [degC] and VPD [kPa] only if/where it is not already
    provided. Equation comes from:

    https://github.com/altazietsman/ML-stomatal-conductance-models/blob/master/Model%20development/BWB.py#L84
    """
    vpd_sat = (
        (
            610.78
            * np.exp(df0[temperature_col] / (df0[temperature_col] + 238.3) * 17.2694)
        )
        / 1000
    ) / 100
    relh = 100 - df0[vpd_col] / vpd_sat
    if rh_col in df0:
        df0[rh_col] = df0[rh_col].fillna(relh)
    else:
        df0[rh_col] = relh
    return df0


def add_cond_ballberry(
    df0: pd.DataFrame,
    photo_col: str,
    rh_col: str,
    ca_col: str,
    cond_col: str,
    gs0: Union[None, float],
    gs1: Union[None, float],
    verbose: bool = False,
) -> pd.DataFrame:
    """Add the BallBerry model of stomatal conductance to the dataframe. If gs0
    or gs1 are specified as None and cond_col is given, then this routine will
    compute the optimal parameters. If gs0 and gs1 are given, then we will
    simply add the model result for those parameters. Model expects
    photosynthesis [umol m-2 s-1], relative humidity [%], and optionally a
    stomatal conductance [mol m-2 s-1]"""

    def _condbb(gs0, gs1, photo, relh, cal):
        return gs0 + gs1 * photo * (0.01 * relh) / cal

    def _residual(gss, photo, relh, cond, cal):
        return np.linalg.norm(_condbb(gss[0], gss[1], photo, relh, cal) - cond)

    if gs0 is None or gs1 is None:
        assert cond_col in df0
        out = minimize(
            _residual,
            [0.0, 15.0],
            (df0[photo_col], df0[rh_col], df0[cond_col], df0[ca_col]),
        )
        if not out.success:
            print(out)
            raise RuntimeError("Optimization of BallBerry parameters failed.")
        if verbose:
            print(f"Optimized BallBerry parameters: gs0 = {out.x[0]}, gs1 = {out.x[1]}")
        df0[f"{cond_col}_bb_opt"] = _condbb(
            out.x[0], out.x[1], df0[photo_col], df0[rh_col], df0[ca_col]
        )
    else:
        df0[f"{cond_col}_bb"] = _condbb(
            gs0, gs1, df0[photo_col], df0[rh_col], df0[ca_col]
        )
    return df0


def add_cond_medlyn(
    df0: pd.DataFrame,
    vpd_col: str,
    photo_col: str,
    ca_col: str,
    cond_col: str,
    gs1: Union[None, float],
    verbose: bool = False,
) -> pd.DataFrame:
    """."""

    def _condm(gs1, vpd, photo, cal):
        return 1.6 * (1 + gs1 / np.sqrt(vpd)) * photo / cal

    def _residual(gss, vpd, photo, cal, cond):
        return np.linalg.norm(_condm(gss[0], vpd, photo, cal) - cond)

    if gs1 is None:
        assert cond_col in df0
        out = minimize(
            _residual,
            [10.0],
            (df0[vpd_col], df0[photo_col], df0[ca_col], df0[cond_col]),
        )
        if not out.success:
            print(out)
            raise RuntimeError("Optimization of Medlyn parameters failed.")
        if verbose:
            print(f"Optimized Medlyn parameter: gs1 = {out.x[0]}")
        df0[f"{cond_col}_m_opt"] = _condm(
            out.x[0], df0[vpd_col], df0[photo_col], df0[ca_col]
        )
    else:
        df0[f"{cond_col}_bb"] = _condm(gs1, df0[vpd_col], df0[photo_col], df0[ca_col])
    return df0


print("\n----------------- Lin2015 ------------------")
df1 = pd.read_csv(
    "./data/Lin2015/WUEdatabase_merged_Lin_et_al_2015_NCC.csv", encoding="ISO-8859-1"
)
df1 = df1.replace(-9999, np.nan)
df1 = df1[["Photo", "Cond", "Tleaf", "VPD", "CO2S"]].dropna()
df1 = remove_outliers(df1, verbose=True)
print("Relative humidity computed from a model")
df1 = add_relative_humidity(
    df1, temperature_col="Tleaf", vpd_col="VPD", rh_col="RH"  # Dataset has no Tair
)
df1 = add_cond_ballberry(df1, "Photo", "RH", "CO2S", "Cond", None, None, verbose=True)
df1 = add_cond_medlyn(df1, "VPD", "Photo", "CO2S", "Cond", None, verbose=True)
print(df1.describe())
print(
    "Optimized BallBerry correlation:",
    np.corrcoef(df1["Cond"], df1["Cond_bb_opt"])[0, 1],
)
print(
    "Optimized Medlyn correlation:",
    np.corrcoef(df1["Cond"], df1["Cond_m_opt"])[0, 1],
)

print("\n---------------- Anderegg2018 ------------------")
df2 = pd.read_csv("./data/Anderegg2018/AllData_EcologyLetters_Figshare_v1_318.csv")
df2 = df2.replace(-9999, np.nan)
df2 = df2[["Photo", "Cond", "Tair", "VPD", "RH", "CO2S"]].dropna()
df2 = remove_outliers(df2, verbose=True)
df2 = add_cond_ballberry(df2, "Photo", "RH", "CO2S", "Cond", None, None, verbose=True)
df2 = add_cond_medlyn(df2, "VPD", "Photo", "CO2S", "Cond", None, verbose=True)
print(df2.describe())
print(
    "Optimized BallBerry correlation:",
    np.corrcoef(df2["Cond"], df2["Cond_bb_opt"])[0, 1],
)
print(
    "Optimized Medlyn correlation:",
    np.corrcoef(df2["Cond"], df2["Cond_m_opt"])[0, 1],
)

print("\n---------------- Saunders ------------------")
print("Photosynthesis added by Farquhar model")
print("Assumed leaf CO2 concentration of 40 ppm as per author script")
df3 = pd.read_pickle("./data/Sauders/Sauders_processed.pkl")
df3 = df3[["Photo", "Cond", "Tair", "VPD", "RH"]].dropna()
df3 = remove_outliers(df3, verbose=True)
df3["ca"] = 40.0  # dataset lacks CO2S and this was constant used in their code
df3 = add_cond_ballberry(df3, "Photo", "RH", "ca", "Cond", None, None, verbose=True)
df3 = add_cond_medlyn(df3, "VPD", "Photo", "ca", "Cond", None, verbose=True)
print(df3.describe())
print(
    "Optimized BallBerry correlation:",
    np.corrcoef(df3["Cond"], df3["Cond_bb_opt"])[0, 1],
)
print(
    "Optimized Medlyn correlation:",
    np.corrcoef(df3["Cond"], df3["Cond_m_opt"])[0, 1],
)
