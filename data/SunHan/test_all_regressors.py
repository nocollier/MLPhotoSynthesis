"""Apply neural networks to SunHan data."""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators


def test_all_regressors(
    x_all: np.ndarray, y_all: np.ndarray, random_state: int = 1
) -> pd.DataFrame:
    """test all the regressors on the input data"""
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, random_state=random_state
    )
    names = []
    passes = []
    scores = []
    times = []
    for name, regressor in all_estimators("regressor"):
        print(name)
        names.append(name)
        tref = time.time()
        try:
            regr = regressor().fit(x_train, y_train)
            passes.append(True)
            times.append(time.time() - tref)
        # pylint: disable=bare-except
        except:
            passes.append(False)
            scores.append(np.nan)
            times.append(np.nan)
            continue

        try:
            score = regr.score(x_test, y_test)
            scores.append(score)
        # pylint: disable=bare-except
        except:
            scores.append(np.nan)
            continue

    return pd.DataFrame({"name": names, "pass": passes, "score": scores, "time": times})


def get_sklearn_regressor(name: str):
    """Get a sklearn regressor by name. Maybe there is a function already?"""
    regrs = [regr for name, regr in all_estimators() if dfrow["name"] == name]
    assert len(regrs) == 1
    return regrs[0]


# read in all the data
df = pd.read_pickle("SunHan.pkl")
df = df.drop(columns=["Species name", "PFT"])
y = df["Anet"].to_numpy().flatten()
df = df.drop(columns="Anet")
X = df.to_numpy()

# test all the regressors and build a dataframe to track performance information
if not os.path.isfile("df_regr.pkl"):
    df_regr = test_all_regressors(X, y)
    df_regr.to_pickle("df_regr.pkl")
df_regr = pd.read_pickle("df_regr.pkl")
df_regr = df_regr.dropna()
df_regr = df_regr[df_regr["time"] < 50]
df_regr = df_regr[df_regr["score"] > 0]
df_regr = df_regr.sort_values("score", ascending=False)

# Now see how the good regressors perform
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
nrow = int(round(np.sqrt(len(df_regr))))
ncol = int(len(df_regr) / nrow)
fig, ax = plt.subplots(figsize=(18, 12), nrows=nrow, ncols=ncol, tight_layout=True)
# pylint: disable=invalid-name
row = 0
col = 0
regressors = {}
for ind, dfrow in df_regr.iterrows():
    regr = get_sklearn_regressor(dfrow["name"])
    regr = regr().fit(x_train, y_train)
    ax[row, col].plot(y_train, regr.predict(x_train), "o")
    ax[row, col].plot(y_test, regr.predict(x_test), "^")
    ax[row, col].set_title(f"{dfrow['name']} {dfrow['score']:.3f}")
    col += 1
    if col == ncol:
        row += 1
        col = 0
    regressors[dfrow["name"]] = regr
fig.savefig("all_regressors.png")
plt.close()

# create a dense random sampling
desc = df.describe()
SAMPLE_SIZE = 10000
X_sample = []
for col in desc.columns:
    X_sample.append(
        desc.loc["min", col]
        + np.random.rand(SAMPLE_SIZE) * (desc.loc["max", col] - desc.loc["min", col])
    )
X_sample = np.vstack(X_sample).T

# now plot Anet vs each column for each regressor
nrow = int(round(np.sqrt(X.shape[1])))
ncol = int(X.shape[1] / nrow)
for ind, dfrow in df_regr.iterrows():
    y_sample = regressors[dfrow["name"]].predict(X_sample)
    fig, ax = plt.subplots(figsize=(18, 12), nrows=nrow, ncols=ncol, tight_layout=True)
    for row in range(nrow):
        for col in range(ncol):
            xcol = ncol * row + col
            if xcol >= X_sample.shape[1]:
                continue
            ax[row, col].plot(X[:, xcol], y, "o")
            ax[row, col].plot(X_sample[:, xcol], y_sample, "ro", ms=1, alpha=0.2)
            ax[row, col].set_ylabel("Anet")
            ax[row, col].set_xlabel(df.columns[xcol])
    fig.suptitle(
        f"{dfrow['name']}, score = {dfrow['score']:.3f}, time = {dfrow['time']:e}"
    )
    fig.savefig(f"{dfrow['name']}.png")
    plt.close()
