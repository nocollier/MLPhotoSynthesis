"""."""
import intake

import photosynthesis as photo

# Read in the data frame, and only use rows where all columns of interest
# contain non-null data
cat = intake.open_catalog(
    "https://raw.githubusercontent.com/nocollier/MLPhotoSynthesis/main/data/leaf-level.yaml"
)
SOURCE = "Anderegg2018"
df = cat[SOURCE].read()
cols = [
    "Tleaf",
    "Cond",
    "CO2S",
    "VPD",
]
if "RH" in df.columns:
    cols.append("RH")
df = df[df[cols].notna().all(axis=1)]

# Add in stomatal conductance models, Ball-Berry needs relative humidity
if "RH" not in df:
    df = photo.add_relative_humidity(
        df, temperature_col="Tleaf", vpd_col="VPD", rh_col="RH"
    )
df = photo.add_cond_ballberry(
    df,
    photo_col="Photo",
    rh_col="RH",
    ca_col="CO2S",
    cond_col="Cond",
    gs0=None,
    gs1=None,
)
df = photo.add_cond_medlyn(
    df,
    photo_col="Photo",
    vpd_col="VPD",
    ca_col="CO2S",
    cond_col="Cond",
    gs0=None,
    gs1=None,
)

# Add in Farquhar model of photosynthesis
VCMAX25 = 83.578991
JMAX25 = 126.914379
df["Photo_Farquhar"] = df.apply(
    photo.apply_photosynthesis_farquhar, axis=1, args=(VCMAX25, JMAX25)
)
df.attrs["notes"].append("Added Farquhar model of photosynthesis")
print("\n".join(df.attrs["notes"]))
print(df)

df.to_pickle(f"df_{SOURCE}.pkl")
