"""Can we demonstrate that ML can do better?"""
import intake
import matplotlib.pyplot as plt
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

# Compute optimal Ball-Berry parameters based on the training data, and then use
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
bb_score = r2_score(df_test["Cond"], df_test["Cond_bb"])

# Now we do the same with random forests.
cols = ["Wregion2", "CO2S", "Trmmol", "PARin", "Tleaf", "Photo", "VPD"]
regr = RandomForestRegressor().fit(df_train[cols], df_train["Cond"])
df_test["Cond_rf"] = regr.predict(df_test[cols])
rf_score = r2_score(df_test["Cond"], df_test["Cond_rf"])

print(bb_score, rf_score)
sns.pairplot(
    df_test[[col for col in df_test.columns if col.startswith("Cond")]],
    diag_kind="kde",
)
plt.show()
