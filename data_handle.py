import io
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

from utils import to_digits


@dataclass
class LotteryAPI:
    base_url: str = "https://example-lottery-api.local"
    game: str = "pick4"

    def fetch_recent_draws(self, limit: int = 500) -> pd.DataFrame:
        # Demo: synthetic draws. Replace with real requests to your API.
        rng = np.random.default_rng(seed=42)
        rows = []
        date = pd.Timestamp.utcnow().normalize()
        cur = (1, 2, 3, 4)
        for i in range(limit):
            shifts = rng.choice([-2, -1, 0, 1, 2], size=4)
            cur = tuple(((cur[j] + int(shifts[j])) % 10) for j in range(4))
            rows.append(
                {
                    "draw_date": date - pd.Timedelta(days=limit - i),
                    "winning_numbers": cur,
                    "sum": int(sum(cur)),
                }
            )
        return pd.DataFrame(rows)


class LotteryDataHandler:
    def __init__(self, api: Optional[LotteryAPI] = None, n_digits: int = 4):
        self.api = api or LotteryAPI()
        self.n_digits = n_digits

    @st.cache_data(show_spinner=False)
    def get_processed_data(self, use_api: bool, uploaded_csv: Optional[bytes]) -> pd.DataFrame:
        if use_api:
            df = self.api.fetch_recent_draws(limit=800)
        else:
            if not uploaded_csv:
                # Small sample if no file provided
                df = pd.DataFrame(
                    {
                        "draw_date": pd.date_range(end=pd.Timestamp.utcnow(), periods=15),
                        "winning_numbers": [
                            (1, 2, 3, 4), (3, 6, 0, 1), (9, 8, 7, 0), (7, 4, 1, 2), (2, 5, 8, 3),
                            (1, 2, 3, 5), (3, 6, 9, 1), (0, 2, 4, 6), (1, 3, 5, 7), (2, 4, 6, 8),
                            (3, 5, 7, 9), (4, 6, 8, 0), (5, 7, 9, 1), (6, 9, 1, 3), (8, 0, 3, 5)
                        ],
                    }
                )
                df["sum"] = df["winning_numbers"].apply(sum)
            else:
                raw = pd.read_csv(io.BytesIO(uploaded_csv))
                if "winning_numbers" in raw.columns and isinstance(raw["winning_numbers"].iloc[0], str):
                    def parse_tuple(s):
                        digs = [int(ch) for ch in s if ch.isdigit()]
                        return tuple(digs[:self.n_digits]) if len(digs) >= self.n_digits else to_digits(digs, self.n_digits)
                    wn = raw["winning_numbers"].apply(parse_tuple)
                else:
                    wn = raw.iloc[:, 0].apply(lambda x: to_digits(x, self.n_digits))

                if "draw_date" in raw.columns:
                    dt = pd.to_datetime(raw["draw_date"], errors="coerce")
                else:
                    dt = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(raw))

                df = pd.DataFrame({"draw_date": dt, "winning_numbers": wn})
                df["sum"] = df["winning_numbers"].apply(sum)

        df = df.dropna(subset=["draw_date"]).sort_values("draw_date").reset_index(drop=True)
        return df
