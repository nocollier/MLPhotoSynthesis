"""Can we demonstrate that ML can do better?"""
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import photosynthesis as ph

cat = intake.open_catalog(
    "https://raw.githubusercontent.com/nocollier/MLPhotoSynthesis/main/data/leaf-level.yaml"
)
src = cat["Lin2015"]
df = src.read()

# Prepare data
df = ph.remove_outliers(
    df,
    columns=["Wregion2", "CO2S", "Trmmol", "PARin", "Tleaf", "Photo", "VPD"],
    verbose=True,
)
df = ph.add_relative_humidity(df, temperature_col="Tleaf", vpd_col="VPD", rh_col="RH")
df_train, df_test = train_test_split(df, train_size=0.8, random_state=12082022)

# Compute optimal Ball-Berry/Medlyn parameters based on the training data, and then use
# the coefficients to apply the model to the test data.
df_train = ph.add_cond_ballberry(
    df_train,
    photo_col="Photo",
    rh_col="RH",
    ca_col="CO2S",
    cond_col="Cond",
    gs0=None,
    gs1=None,
)
df_test = ph.add_cond_ballberry(
    df_test,
    photo_col="Photo",
    rh_col="RH",
    ca_col="CO2S",
    cond_col="Cond",
    gs0=df_train.attrs["Ball-Berry"]["gs0"],
    gs1=df_train.attrs["Ball-Berry"]["gs1"],
)
df_train = ph.add_cond_medlyn(
    df_train,
    photo_col="Photo",
    vpd_col="VPD",
    ca_col="CO2S",
    cond_col="Cond",
    gs0=None,
    gs1=None,
)
df_test = ph.add_cond_medlyn(
    df_test,
    photo_col="Photo",
    vpd_col="VPD",
    ca_col="CO2S",
    cond_col="Cond",
    gs0=df_train.attrs["Medlyn"]["gs0"],
    gs1=df_train.attrs["Medlyn"]["gs1"],
)
bb_score = r2_score(df_test["Cond"], df_test["Cond_bb"])
m_score = r2_score(df_test["Cond"], df_test["Cond_m"])

# Now we do the same with random forests, one uses just the columns also in the
# Medlyn model, the other uses all the nyumerical columns we have.
cols = ["Wregion2", "CO2S", "Trmmol", "PARin", "Tleaf", "Photo", "VPD"]
cols_m = ["CO2S", "Photo", "VPD"]

regr_m = RandomForestRegressor().fit(df_train[cols_m], df_train["Cond"])
df_test["Cond_rf_m"] = regr_m.predict(df_test[cols_m])
rf_m_score = r2_score(df_test["Cond"], df_test["Cond_rf_m"])

regr_all = RandomForestRegressor().fit(df_train[cols], df_train["Cond"])
df_test["Cond_rf"] = regr_all.predict(df_test[cols])
rf_score = r2_score(df_test["Cond"], df_test["Cond_rf"])

dfi = pd.DataFrame(
    {
        "features": regr_all.feature_names_in_,
        "importance": regr_all.feature_importances_,
    }
)
print(dfi.sort_values("importance", ascending=False))

# Plot a comparison of each method with respect to the measured data.
plt.rcParams.update({"font.size": 16})
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 14), tight_layout=True)
fig.subplots_adjust(hspace=0.125, wspace=0.125)
sns.despine(fig, left=True, bottom=True)
cmax = df_test[[c for c in df.columns if c.startswith("Cond")]].to_numpy().max()
pad = 0.05 * cmax
for ind, col in enumerate(["Cond_bb", "Cond_m", "Cond_rf_m", "Cond_rf"]):
    i = int(ind / 2)
    j = ind % 2
    sns.scatterplot(
        x="Cond",
        y=col,
        hue="Photo",
        hue_order=df_test["Photo"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]),
        linewidth=0,
        data=df_test,
        ax=axs[i, j],
        legend="auto" if ind == 0 else False,
    )
    axs[i, j].set_xticks(np.linspace(0, 1.25, 6))
    axs[i, j].set_yticks(np.linspace(0, 1.25, 6))
    axs[i, j].set_xlim(-pad, cmax + pad)
    axs[i, j].set_ylim(-pad, cmax + pad)
axs[0, 0].set_title(f"Ball-Berry, $R^2$ = {bb_score:.3f}")
axs[0, 1].set_title(f"Medlyn, $R^2$ = {m_score:.3f}")
axs[1, 0].set_title(f"Random Forest (Medlyn), $R^2$ = {rf_m_score:.3f}")
axs[1, 1].set_title(f"Random Forest, $R^2$ = {rf_score:.3f}")
fig.savefig("ml_win.png")
plt.close()

# Densely sample the regressor along the following axes, other columns are set
# to the mean values from the observational data
pcols = ["Photo", "VPD"]  # which 2 columns to plot
N1D = 1000
arrays = ph.create_latin_hypercube(df[pcols], N1D)
dflhc = pd.DataFrame({pcols[i]: arrays[i].flatten() for i in range(len(pcols))})
for col in cols:
    if col in pcols:
        continue
    dflhc[col] = float(df[col].mean())
dflhc["Cond_rf"] = regr_all.predict(dflhc[cols])
dflhc = ph.add_cond_medlyn(
    dflhc,
    photo_col="Photo",
    vpd_col="VPD",
    ca_col="CO2S",
    cond_col="Cond",
    gs0=df_train.attrs["Medlyn"]["gs0"],
    gs1=df_train.attrs["Medlyn"]["gs1"],
)

# Create a 2 panel comparison plot
fig = plt.figure(constrained_layout=True, figsize=(13, 7.7))
subfigs = fig.subfigures(1, 1)
axs = subfigs.subplots(1, 2, sharey=True)
vmin = min(dflhc["Cond_m"].min(), dflhc["Cond_rf"].min())
vmax = max(dflhc["Cond_m"].max(), dflhc["Cond_rf"].max())
qax = axs[0].pcolormesh(
    dflhc[pcols[0]].to_numpy().reshape((N1D, N1D)),
    dflhc[pcols[1]].to_numpy().reshape((N1D, N1D)),
    dflhc["Cond_m"].to_numpy().reshape((N1D, N1D)),
    vmin=vmin,
    vmax=vmax,
)
pax = axs[1].pcolormesh(
    dflhc[pcols[0]].to_numpy().reshape((N1D, N1D)),
    dflhc[pcols[1]].to_numpy().reshape((N1D, N1D)),
    dflhc["Cond_rf"].to_numpy().reshape((N1D, N1D)),
    vmin=vmin,
    vmax=vmax,
)
subfigs.colorbar(
    pax, shrink=0.6, ax=axs, location="bottom", label="Stomatal Conductance"
)
axs[0].set_title("Medlyn")
axs[1].set_title("Random Forest")
for i in range(2):
    axs[i].set_xlabel(pcols[0])
    if i == 0:
        axs[i].set_ylabel(pcols[1])
fig.savefig("nonsmooth.png")
plt.close()
