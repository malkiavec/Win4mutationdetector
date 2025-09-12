from typing import Tuple

import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, n_digits: int = 4):
        self.n_digits = n_digits
        self.fitted_ = False
        self.means_ = None
        self.stds_ = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        digs = pd.DataFrame(df["winning_numbers"].tolist(), columns=[f"d{i+1}" for i in range(self.n_digits)])
        out = pd.DataFrame()
        out[[f"d{i+1}" for i in range(self.n_digits)]] = digs
        out["sum"] = digs.sum(axis=1)
        out["odd_count"] = digs.apply(lambda r: sum(1 for v in r if v % 2 == 1), axis=1)
        out["even_count"] = self.n_digits - out["odd_count"]
        out["unique_count"] = digs.apply(lambda r: len(set(r)), axis=1)
        out["repeat_pair"] = digs.apply(lambda r: int(any(c >= 2 for c in r.value_counts().values)), axis=1)
        out["sum_rolling_mean_10"] = out["sum"].rolling(10, min_periods=1).mean()
        out["sum_rolling_std_10"] = out["sum"].rolling(10, min_periods=1).std().fillna(0.0)
        return out

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feats = self.create_features(df)
        X_train = feats.iloc[:-1].reset_index(drop=True)
        X_test = feats.iloc[[-1]].reset_index(drop=True)
        self.fit_scaler(X_train)
        return self.transform_features(X_train), self.transform_features(X_test)

    def fit_scaler(self, X: pd.DataFrame):
        self.means_ = X.mean()
        self.stds_ = X.std().replace(0, 1.0)
        self.fitted_ = True

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            self.fit_scaler(X)
        return (X - self.means_) / self.stds_
