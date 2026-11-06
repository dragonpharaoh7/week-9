import pickle

import pandas as pd


class GroupEstimate:
    """Map categorical input to a continuous estimate, learned from data."""

    def __init__(self, estimate="mean"):
        if estimate not in ("mean", "median"):
            raise ValueError(f"estimate must be 'mean' or 'median', got {estimate!r}")
        self.estimate = estimate

    def fit(self, X, y):
        """Store the mean/median of `y` for each group defined by `X`."""
        df = pd.DataFrame(X).reset_index(drop=True)
        self.columns_ = list(df.columns)
        df["_y"] = pd.Series(y).reset_index(drop=True)

        grouped = df.groupby(self.columns_, observed=True)["_y"]
        estimates = grouped.mean() if self.estimate == "mean" else grouped.median()
        self.estimates_ = estimates.rename("estimate").reset_index()

        return self

    def predict(self, X_):
        """Return the estimate for each observation, or NaN if its group is new."""
        df = pd.DataFrame(X_)
        df.columns = self.columns_

        y_ = df.merge(self.estimates_, on=self.columns_, how="left")["estimate"]

        n_missing = y_.isna().sum()
        if n_missing:
            print(f"{n_missing} observation(s) had no matching group.")

        return y_.to_numpy()

    def save(self, path):
        """Pickle the fitted model so an app can load it instead of refitting."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Load a model written by `.save()`."""
        with open(path, "rb") as f:
            return pickle.load(f)
