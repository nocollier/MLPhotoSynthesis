"""Read"""
import os

import pandas as pd
import plotly.express as px

import intake

if os.path.isfile("SunHan.pkl"):
    df = pd.read_pickle("SunHan.pkl")
else:
    cat = intake.open_catalog("SunHan.yaml")
    df_all = []
    for src in cat:
        print(src)
        df = cat[src].read()
        df["PFT"] = src
        df = df.rename({col: col.strip() for col in df.columns}, axis=1)
        df_all.append(df)
    df = pd.concat(df_all)
    df.to_pickle("SunHan.pkl")
    df.to_csv("SunHan.csv")

fig = px.scatter(df, x="PARi", y="Anet", color="PFT")
fig.show()
