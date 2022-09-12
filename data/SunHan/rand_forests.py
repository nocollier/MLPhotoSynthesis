"""Experimenting with random forests applied to the photosynthesis data."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load up the data and choose data
df = pd.read_pickle("SunHan.pkl")
df = df.drop(columns=["Species name", "PFT"])
y = df["Anet"].to_numpy().flatten()
df = df.drop(columns="Anet")
X = df.to_numpy()

# Split into training and test sets and build forest
X_train, X_test, y_train, y_test = train_test_split(X, y)
regr = RandomForestRegressor().fit(X_train, y_train)
y_pred = regr.predict(X_test)
score = regr.score(X_test, y_test)
print("Feature Importances:")
for c, i in zip(df.columns, regr.feature_importances_):
    print(f"{i:.3f} {c}")
print(f"Score: {score:.3f}")

# Plot Anet vs 'column whose importance is > 10%'
IMPORTANCE_THRESHOLD = 0.1
ncols = (regr.feature_importances_ > IMPORTANCE_THRESHOLD).sum()
fig, ax = plt.subplots(ncols=ncols, tight_layout=True)
for i, col in zip(
    range(ncols), np.where(regr.feature_importances_ > IMPORTANCE_THRESHOLD)[0]
):
    ax[i].plot(X_test[:, col], y_test, "o", label="Test data")
    ax[i].plot(X_test[:, col], y_pred, "^", label="Predicted values")
    ax[i].set_ylabel("Anet")
    ax[i].set_xlabel(df.columns[col])
    ax[i].legend()
fig.suptitle("Random Forest Out of Sample Comparison for Most Important Features")
plt.show()

# Plot Anet vs fabricated data from range of inputs
SAMPLE_SIZE = 10000
df_info = df.describe()
X_sample = []
for col in df_info.columns:
    X_sample.append(
        df_info.loc["min", col]
        + np.random.rand(SAMPLE_SIZE)
        * (df_info.loc["max", col] - df_info.loc["min", col])
    )
X_sample = np.vstack(X_sample).T
y_sample = regr.predict(X_sample)
fig, ax = plt.subplots(ncols=ncols, tight_layout=True)
for i, col in zip(
    range(ncols), np.where(regr.feature_importances_ > IMPORTANCE_THRESHOLD)[0]
):
    ax[i].plot(X[:, col], y, "o", label="Original data")
    ax[i].plot(X_sample[:, col], y_sample, "ro", ms=1, label="Sampled")
    ax[i].set_ylabel("Anet")
    ax[i].set_xlabel(df.columns[col])
    ax[i].legend()
fig.suptitle("Random Forest Uniformly Sampled for Important Features")
plt.show()
