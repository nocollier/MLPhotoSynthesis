"""Apply neural networks to SunHan data."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators

df = pd.read_pickle("SunHan.pkl")
df = df[df["PFT"] == "Andropogon_gerardii"]
df = df[["Anet", "Tleaf", "PARi", "CO2R", "Ci"]]

X = df.to_numpy()[:, 1:]
y = df.to_numpy()[:, 0]

all_est = [
    e
    for e in all_estimators("regressor")
    if e[0]
    not in [
        "55",
        "CCA",
        "GammaRegressor",
        "IsotonicRegression",
        "MultiOutputRegressor",
        "MultiTaskElasticNet",
        "MultiTaskElasticNetCV",
        "MultiTaskLasso",
        "MultiTaskLassoCV",
        "PLSCanonical",
        "PoissonRegressor",
        "RegressorChain",
        "StackingRegressor",
        "VotingRegressor",
    ]
]
ntot = len(all_est)
nrow = int(round(np.sqrt(ntot)))
ncol = int(ntot / nrow)
fig, ax = plt.subplots(nrows=nrow, ncols=ncol, tight_layout=True)
c = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(y_test)
for name, Regressor in all_est:
    c += 1
    i = int(c / ncol)
    j = c - i * ncol
    regr = Regressor().fit(X_train, y_train)
    try:
        s = regr.score(X_test, y_test)
    except:
        s = 0
    ax[i, j].plot(y_train, regr.predict(X_train), "o")
    ax[i, j].plot(y_test, regr.predict(X_test), "^")
    ax[i, j].set_title(f"{name} {max(0,s):.3f}")
plt.show()
