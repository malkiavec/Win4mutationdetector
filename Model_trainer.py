import random
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np

from utils import multiset_overlap, greedy_multiset_mapping, load_boosts, save_boosts, to_digits


class LotteryModelTrainer:
    def __init__(self, n_digits: int = 4):
        self.n_digits = n_digits
        self.alpha_ = 0.5  # smoothing
        self.digit_freq_ = np.ones(10)
        self.trans_counts_ = Counter()

    def fit(self, history_numbers: List[Tuple[int, ...]]):
        # Digit frequencies
        freq = Counter()
        for t in history_numbers:
            for d in t:
                freq[d] += 1
        self.digit_freq_ = np.array([freq.get(i, 0) + 1.0 for i in range(10)], dtype=float)
        self.digit_freq_ = self.digit_freq_ / self.digit_freq_.sum()

        # Positionless transitions (lag=1)
        self.trans_counts_.clear()
        for i in range(len(history_numbers) - 1):
            a, b = history_numbers[i], history_numbers[i + 1]
            for x, y in greedy_multiset_mapping(a, b):
                self.trans_counts_[(x, y)] += 1

    def transition_probs(self, boosts: Optional[Dict[str, float]] = None, boost_weight: float = 1.0) -> Dict[Tuple[int, int], float]:
        cnt = self.trans_counts_.copy()
        totals = Counter()
        for (x, y), c in cnt.items():
            totals[x] += c
        if boosts:
            for k, v in boosts.items():
                try:
                    xs, ys = k.split("->")
                    x, y = int(xs), int(ys)
                    cnt[(x, y)] += float(v) * boost_weight
                    totals[x] += float(v) * boost_weight
                except Exception:
                    continue
        probs: Dict[Tuple[int, int], float] = {}
        alpha = self.alpha_
        for x in range(10):
            denom = totals[x] + alpha * 10
            for y in range(10):
                c = cnt.get((x, y), 0.0)
                probs[(x, y)] = (c + alpha) / (denom if denom > 0 else 10.0)
        return probs

    def base_predict_distribution(self, last_draw: Tuple[int, ...], boosts: Optional[Dict[str, float]] = None, boost_weight: float = 1.0) -> List[Tuple[int, float]]:
        probs = self.transition_probs(boosts=boosts, boost_weight=boost_weight)
        per_digit = []
        for d in last_draw:
            row = [(y, probs.get((d, y), 1e-3)) for y in range(10)]
            row.sort(key=lambda t: t[1], reverse=True)
            per_digit.append(row[:4])  # top-4 per digit
        cands: Dict[Tuple[int, ...], float] = {}
        def dfs(i, cur, p):
            if i == len(per_digit):
                t = tuple(cur)
                cands[t] = cands.get(t, 0.0) + p
                return
            for y, py in per_digit[i]:
                cur.append(y)
                dfs(i + 1, cur, p * py)
                cur.pop()
        dfs(0, [], 1.0)
        total_p = sum(cands.values()) or 1.0
        ranked = sorted(((t, v / total_p) for t, v in cands.items()), key=lambda kv: kv[1], reverse=True)
        return ranked

    def mutate(self, t: Tuple[int, ...], strength: float) -> Tuple[int, ...]:
        max_step = max(1, int(5 * strength))
        return tuple((d + random.randint(-max_step, max_step)) % 10 for d in t)

    def generate_predictions(self, last_draw: Tuple[int, ...], n: int, mutation_strength: float,
                             boosts: Optional[Dict[str, float]] = None, boost_weight: float = 1.0) -> List[Tuple[int, ...]]:
        ranked = self.base_predict_distribution(last_draw, boosts=boosts, boost_weight=boost_weight)
        seeds = [t for t, _ in ranked[:max(5, n//2)]]
        preds = set(seeds)
        while len(preds) < n:
            base = random.choice(seeds)
            preds.add(self.mutate(base, mutation_strength))
        return list(preds)[:n]

    def mark_success_with_boosts(self, seed: Tuple[int, ...], best_pred: Tuple[int, ...], enable_boosts: bool, boost_step: float = 1.0):
        if not enable_boosts:
            return
        pairs = greedy_multiset_mapping(seed, best_pred)
        b = load_boosts()
        for x, y in pairs:
            key = f"{x}->{y}"
            b[key] = b.get(key, 0.0) + float(boost_step)
        save_boosts(b)

    def evaluate_sequence(self, history_numbers: List[Tuple[int, ...]], n_preds: int = 30,
                          mutation_strength: float = 0.3, boosts: Optional[Dict[str, float]] = None,
                          boost_weight: float = 1.0) -> List[Dict]:
        rows = []
        for t in range(1, len(history_numbers)):
            train_hist = history_numbers[:t]
            self.fit(train_hist[:-1] if len(train_hist) > 1 else train_hist)
            last = train_hist[-1]
            preds = self.generate_predictions(last, n=n_preds, mutation_strength=mutation_strength,
                                              boosts=boosts, boost_weight=boost_weight)
            actual = history_numbers[t]
            best_overlap = max(multiset_overlap(p, actual) for p in preds) if preds else 0
            rows.append({"t": t, "best_overlap": best_overlap, "success3": int(best_overlap >= 3), "success4": int(best_overlap >= 4)})
        return rows
