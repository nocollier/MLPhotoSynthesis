"""Routines for testing performance of stomatal conductance models on data.
"""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def remove_outliers(
    df0: pd.DataFrame, columns: List = None, verbose: bool = False, outlier: float = 3.0
) -> pd.DataFrame:
    """Removes rows where any column has data greater than 3 standard deviations
    from the mean. If 'columns' are specified, only include these in the
    calculation."""
    df_include = df0 if columns is None else df0[columns]
    df_reduced = df0[
        ((np.abs(df_include - df_include.mean()) / df_include.std()) < outlier).all(
            axis=1
        )
    ]
    nrm = len(df0) - len(df_reduced)
    if verbose:
        print(f"Removed {nrm} rows ({100*nrm/len(df0):.1f}%) marked as outliers.")
    return df_reduced


def create_latin_hypercube(dfin: pd.DataFrame, n1d: int) -> Tuple[np.array]:
    """."""
    dfs = dfin.describe().transpose()
    arrays = [np.linspace(row["25%"], row["75%"], n1d) for _, row in dfs.iterrows()]
    return np.meshgrid(*arrays)


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
    if "notes" not in df0.attrs:
        df0.attrs["notes"] = []
    df0.attrs["notes"].append(f"Added relative humidity '{rh_col}' using an equation")
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
    """Add the Ball-Berry model of stomatal conductance to the dataframe.

    Given the column names for

    * photosynthesis (photo_col) [umol m-2 s-1],
    * relative humidity (rh_col) [%], and
    * the CO2 concentration at the leaf surface (ca_col) [ppm]

    compute the stomatal conductance (cond_col) [mol m-2 s-1] by the Ball-Berry
    model using the input parameters gs0 [mol m-2 s-1] and gs1 [%-1]. The model
    will be added to the dataframe in a column labeled cond_col with '_bb'
    postpended. If the constants gs0 and gs1 are given as None and cond_col is
    already in the dataframe, then we will compute optimal parameters and add
    the result to the dataframe with the column name cond_col with '_bb_opt'
    postpended.
    """

    def _condbb(gs0, gs1, photo, relh, cal):
        return gs0 + gs1 * photo * (0.01 * relh) / cal

    def _residual(gss, photo, relh, cond, cal):
        return np.linalg.norm(_condbb(gss[0], gss[1], photo, relh, cal) - cond)

    if "notes" not in df0.attrs:
        df0.attrs["notes"] = []
    if gs0 is None or gs1 is None:
        assert cond_col in df0
        out = minimize(
            _residual,
            [0.0, 15.0],  # use typical values as an initial guess
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
        df0.attrs["Ball-Berry"] = {"gs0": out.x[0], "gs1": out.x[1]}
        df0.attrs["notes"].append(
            f"Added optimized Ball-Berry model '{cond_col}_bb_opt' using gs0={out.x[0]} and gs1={out.x[1]}"
        )
    else:
        df0[f"{cond_col}_bb"] = _condbb(
            gs0, gs1, df0[photo_col], df0[rh_col], df0[ca_col]
        )
        df0.attrs["notes"].append(
            f"Added Ball-Berry model '{cond_col}_bb' using {gs0=} and {gs1=}"
        )
    return df0


def add_cond_medlyn(
    df0: pd.DataFrame,
    vpd_col: str,
    photo_col: str,
    ca_col: str,
    cond_col: str,
    gs0: Union[None, float],
    gs1: Union[None, float],
    verbose: bool = False,
) -> pd.DataFrame:
    """Add the Medlyn model of stomatal conductance to the dataframe.

    Given the column names for

    * leaf boundary layer vapor pressure deficit (vpd_col) [kPa],
    * photosynthesis (photo_col) [umol m-2 s-1], and
    * the CO2 concentration at the leaf surface (ca_col) [ppm]

    compute the stomatal conductance (cond_col) [mol m-2 s-1] by the Medlyn
    model using the input parameters gs0 [mol m-2 s-1] and gs1 [1]. The model
    will be added to the dataframe in a column labeled cond_col with '_m'
    postpended. If the constants gs0 and gs1 are given as None and cond_col is
    already in the dataframe, then we will compute optimal parameters and add
    the result to the dataframe with the column name cond_col with '_m_opt'
    postpended.
    """

    def _condm(gs0, gs1, vpd, photo, cal):
        return gs0 + 1.6 * (1 + gs1 / np.sqrt(vpd)) * photo / cal

    def _residual(gss, vpd, photo, cal, cond):
        return np.linalg.norm(_condm(gss[0], gss[1], vpd, photo, cal) - cond)

    if "notes" not in df0.attrs:
        df0.attrs["notes"] = []
    if gs0 is None or gs1 is None:
        assert cond_col in df0
        out = minimize(
            _residual,
            [0, 10.0],
            (df0[vpd_col], df0[photo_col], df0[ca_col], df0[cond_col]),
        )
        if not out.success:
            print(out)
            raise RuntimeError("Optimization of Medlyn parameters failed.")
        if verbose:
            print(f"Optimized Medlyn parameter: gs0 = {out.x[0]} gs1 = {out.x[1]}")
        df0[f"{cond_col}_m_opt"] = _condm(
            out.x[0], out.x[1], df0[vpd_col], df0[photo_col], df0[ca_col]
        )
        df0.attrs["Medlyn"] = {"gs0": out.x[0], "gs1": out.x[1]}
        df0.attrs["notes"].append(
            f"Added optimized Medlyn model '{cond_col}_m_opt' using gs0={out.x[0]} and gs1={out.x[1]}"
        )
    else:
        df0[f"{cond_col}_m"] = _condm(
            gs0, gs1, df0[vpd_col], df0[photo_col], df0[ca_col]
        )
        df0.attrs["notes"].append(
            f"Added Medlyn model '{cond_col}_bb' using {gs0=} and {gs1=}"
        )
    return df0


# pylint: disable=invalid-name
def _quadratic(a, b, c):
    """Solve a quadratic of the form a x^2 + b x + c = 0"""
    assert (4 * a * c) <= (b**2)
    tmp = np.sqrt(b**2 - 4 * a * c)
    return (-b + tmp) / (2 * a), (-b - tmp) / (2 * a)


def apply_photosynthesis_farquhar(row, Vcmax25=42.0, Jmax25=65.0):
    """
    https://escomp.github.io/ctsm-docs/versions/release-clm5.0/html/tech_note/Photosynthesis/CLM50_Tech_Note_Photosynthesis.html

    * I have only used expressions from the C3 pathway, do we need to
      distinguish? Is our data that specific (Lin2015 does have C3/C4)?
        - C4 equations didn't work out, Aj was always 2 orders of magnitude
          smaller than the other rates, feels like I am doing something wrong.
    * In computing Aj and Ac (2.9.3 & 2.9.4) I have interpretted the ci >= Gamma
      as max(ci-Gamma,0). This means that if ci < Gamma, the net photosynthesis
      will be the respiration with a negative sign.
    * The nonlinear solver fails 85% of the time. I have tried different
      solvers, but the Hessian is not SPD. I could use PETSc SNES, but it may be
      that the solver complains because of the min statement in the residual.
      Try the smoothing mentioned in the MAAT paper.
        - Adding in the Jacobian kills convergence. Not sure why.
        - Smoothing doesn't help.
        - minimize is reporting success=False most of the time, but the the
          function is being reduced 8 to 11 orders of magnitude which is more
          than ample.
    """
    # Constants
    Rgas = 8.31446  # [J mol-1 K-1]
    Tref = 25 + 273.15  # [K]
    Patm = 101325.0  # [Pa]
    oi = 0.2 * Patm  # [Pa]
    Kc25 = 404.9e-6 * Patm  # [Pa]
    Ko25 = 278.4e-3 * Patm  # [Pa]
    Gamma25 = 42.75e-6 * Patm  # CO2 compensation point [Pa]
    # Vcmax25 = 42  # ??? [umol m-2 s-1], Nate found 35, Elias' default 60
    # Jmax25 = 65  # ??? [umol m-2 s-1]
    Rd25 = 0.015 * Vcmax25

    # Variables that are coming from the data
    if row[["Tleaf", "CO2S", "PARin", "Cond"]].isna().any():
        return np.nan
    Tleaf = row["Tleaf"] + 273.15  # [degC] to [K]
    cs = row["CO2S"] * 1e-6 * Patm  # [ppm] to [Pa] CO2 partial pressure at leaf
    phi = row["PARin"]  # [umol m-2 s-1] photosynthetically active radiation
    gs = row["Cond"]  # [mol m-2 s-1] stomatal conductance

    # Temperature scaling, constants found in Table 2.9.2. The additional
    # scaling for Vcmax, Jmax, and Rd involve a 10-day mean temperature which we
    # do not have for this data.
    def _temperature_scaling(Tv, Ha):
        """Tv [K], Ha [J mol-1]"""
        return np.exp(Ha / (Tref * Rgas) * (1 - Tref / Tv))

    Vcmax = Vcmax25 * _temperature_scaling(Tleaf, 72000)
    Jmax = Jmax25 * _temperature_scaling(Tleaf, 50000)
    Rd = Rd25 * _temperature_scaling(Tleaf, 46390)
    Kc = Kc25 * _temperature_scaling(Tleaf, 79430)
    Ko = Ko25 * _temperature_scaling(Tleaf, 36380)
    Gamma = Gamma25 * _temperature_scaling(Tleaf, 37830)

    # Solve (2.9.6) for electron transport rate
    Theta_PSII = 0.7
    Phi_PSII = 0.85
    I_PSII = 0.5 * Phi_PSII * phi
    Jx = min(*_quadratic(Theta_PSII, -(I_PSII + Jmax), I_PSII * Jmax))  # [umol m-2 s-1]

    # Solve simultanesouly for (ci, An)
    def _jacobian(x):
        ci = x[0]
        An = x[1]
        Aj = Jx * max(ci - Gamma, 0) / (4 * ci + 8 * Gamma)  # (2.9.4)
        Ac = Vcmax * max(ci - Gamma, 0) / (ci + Kc * (1 + oi / Ko))  # (2.9.3)
        Ap = 0.5 * Vcmax  # (2.9.5)
        res = [
            An - ((cs - ci) / (1.6 * Patm * 1e-6) * gs),  # (2.9.19)
            An - (min(Aj, Ac, Ap) - Rd),  # (2.9.2)
        ]
        return res

    def _function(x):
        return np.linalg.norm(_jacobian(x))

    out = minimize(
        _function,
        [cs, 10.0],
    )

    # if we do not reduce the function by at least 6 orders of magnitude, return
    # nan
    if np.log10(_function([cs, 10.0]) / out.fun) < 6:
        return np.nan

    return out.x[1]
