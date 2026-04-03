"""Green vs dirty month sequences from project CSVs (inputs are read-only)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
POWER_CSV = PROJECT_ROOT / "data_final/03_power_generation/electric_power_operations_monthly_all_states.csv"
SEDS_CSV = PROJECT_ROOT / "data_final/06_seds_state_energy/complete_seds_all_energy_1960_2023.csv"

SEQ_LEN = 6
# fos_share, nuc_share, month_sin, month_cos, log1p(all_gen), tetce_z — no ren_share (avoids label leakage)
N_FEATURES = 6


def _state_codes_from_power(df: pd.DataFrame) -> list[str]:
    return sorted(
        s for s in df["location"].unique()
        if len(str(s)) == 2 and str(s).isalpha() and s != "US"
    )


def load_state_tetce(year: int = 2023) -> pd.DataFrame:
    sed = pd.read_csv(SEDS_CSV, usecols=["MSN", "StateCode", "Year", "Data"])
    sed = sed[(sed["MSN"] == "TETCE") & (sed["Year"] == year)]
    return sed.rename(columns={"Data": "tetce_mmt", "StateCode": "state"})[["state", "tetce_mmt"]]


def load_power_state_month_table() -> pd.DataFrame:
    df = pd.read_csv(
        POWER_CSV,
        usecols=["period", "location", "sectorid", "fueltypeid", "generation"],
    )
    states = _state_codes_from_power(df)
    df = df[
        df["location"].isin(states)
        & (df["sectorid"] == 1)
        & (df["fueltypeid"].isin(["ALL", "FOS", "REN", "NUC"]))
    ].copy()
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce").fillna(0.0)
    piv = df.pivot_table(index=["period", "location"], columns="fueltypeid", values="generation", aggfunc="sum")
    for c in ["FOS", "REN", "NUC"]:
        if c not in piv.columns:
            piv[c] = 0.0
        piv[c] = piv[c].fillna(0.0)
    piv["ALL"] = piv["ALL"].fillna(0.0)
    piv = piv[piv["ALL"] > 0].copy()
    piv["ren_share"] = (piv["REN"] / piv["ALL"]).clip(0.0, 1.0)
    piv["fos_share"] = (piv["FOS"] / piv["ALL"]).clip(0.0, 1.0)
    piv["nuc_share"] = (piv["NUC"] / piv["ALL"]).clip(-0.01, 1.0)
    piv = piv.reset_index()
    piv["period"] = pd.to_datetime(piv["period"])
    piv["month"] = piv["period"].dt.month
    piv["year"] = piv["period"].dt.year
    return piv


def monthly_top_green_states(
    top_n: int = 5,
    also_include: tuple[str, ...] = ("TX", "CA", "VA", "AZ"),
    exclude_states: tuple[str, ...] = ("VT", "MT", "IA"),
) -> pd.DataFrame:
    """
    For each calendar month, take the top ``top_n`` states by **renewable share**
    among states **not** in ``exclude_states``, then add any **missing** states from
    ``also_include`` when that month has data for them. Rows are merged, deduped
    by state, sorted by ``ren_share`` (greenest first), and re-ranked.

    Pass ``also_include=()`` to only show the top ``top_n`` states (after exclusions).
    Pass ``exclude_states=()`` to rank across all states.

    Columns: ``period``, ``year``, ``month``, ``month_label``, ``rank``, ``state``,
    ``ren_share``, ``ren_share_pct``, ``in_top_n_slice`` (whether the state was in
    the initial top-``top_n`` cut before merging extras).
    """
    df = load_power_state_month_table()
    extras = tuple(s.upper().strip() for s in also_include)
    excluded = tuple(s.upper().strip() for s in exclude_states)
    excluded_set = set(excluded)
    rows: list[dict] = []
    for period in sorted(df["period"].unique()):
        month_all = df[df["period"] == period]
        month_rank = month_all[~month_all["location"].isin(excluded)]
        top = month_rank.sort_values("ren_share", ascending=False).head(top_n)
        in_top = set(top["location"].astype(str))
        add_frames = [top]
        for st in extras:
            if st in excluded_set:
                continue
            if st not in in_top:
                hit = month_all[month_all["location"] == st]
                if len(hit) > 0:
                    add_frames.append(hit.iloc[:1])
        chunk = pd.concat(add_frames, ignore_index=True) if len(add_frames) > 1 else top
        chunk = chunk.drop_duplicates(subset=["location"], keep="first")
        chunk = chunk.sort_values("ren_share", ascending=False).reset_index(drop=True)
        top_set = set(top["location"].astype(str))
        ts = pd.Timestamp(period)
        for rank, (_, r) in enumerate(chunk.iterrows(), start=1):
            st = str(r["location"])
            rs = float(r["ren_share"])
            rows.append(
                {
                    "period": period,
                    "year": int(ts.year),
                    "month": int(ts.month),
                    "month_label": ts.strftime("%Y-%m"),
                    "rank": rank,
                    "state": st,
                    "ren_share": rs,
                    "ren_share_pct": 100.0 * rs,
                    "in_top_n_slice": st in top_set,
                }
            )
    return pd.DataFrame(rows)


def _assign_green_labels(df: pd.DataFrame) -> pd.Series:
    med_state = df.groupby("location")["ren_share"].transform("median")
    counts = df.groupby("location")["ren_share"].transform("count")
    global_med = df["ren_share"].median()
    threshold = np.where(counts >= 6, med_state, global_med)
    return (df["ren_share"] >= threshold).astype(np.int64)


def build_labeled_frame(co2_year: int = 2023) -> pd.DataFrame:
    df = load_power_state_month_table()
    co2 = load_state_tetce(year=co2_year)
    co2["tetce_z"] = (co2["tetce_mmt"] - co2["tetce_mmt"].mean()) / co2["tetce_mmt"].std(ddof=0)
    df = df.merge(
        co2.rename(columns={"state": "location"})[["location", "tetce_mmt", "tetce_z"]],
        on="location",
        how="left",
    )
    df["tetce_z"] = df["tetce_z"].fillna(0.0)
    df["label_green"] = _assign_green_labels(df)
    return df


def _month_features(row: pd.Series) -> np.ndarray:
    m = int(row["month"])
    ang = 2 * np.pi * (m - 1) / 12.0
    return np.array(
        [
            float(row["fos_share"]),
            float(row["nuc_share"]),
            np.sin(ang),
            np.cos(ang),
            np.log1p(float(row["ALL"])),
            float(row["tetce_z"]),
        ],
        dtype=np.float32,
    )


def build_sequence_tensors(df: pd.DataFrame):
    states = sorted(df["location"].unique())
    st_to_i = {s: i for i, s in enumerate(states)}
    rows, masks, ys, meta_rows = [], [], [], []

    for loc, g in df.groupby("location"):
        g = g.sort_values("period")
        idx_order = g.index.tolist()
        periods = g["period"].tolist()
        labels = g["label_green"].tolist()
        feats = [_month_features(g.loc[i]) for i in idx_order]

        for j in range(len(feats)):
            start = max(0, j - (SEQ_LEN - 1))
            window = feats[start : j + 1]
            pad_len = SEQ_LEN - len(window)
            if pad_len > 0:
                pad = [np.zeros(N_FEATURES, dtype=np.float32) for _ in range(pad_len)]
                seq = pad + window
                mask = [True] * pad_len + [False] * len(window)
            else:
                seq = window[-SEQ_LEN:]
                mask = [False] * SEQ_LEN
            rows.append(np.stack(seq, axis=0))
            masks.append(np.array(mask, dtype=np.bool_))
            ys.append(int(labels[j]))
            meta_rows.append({"location": loc, "period": periods[j], "state_idx": st_to_i[loc]})

    X = np.stack(rows, axis=0)
    mask_arr = np.stack(masks, axis=0)
    y = np.array(ys, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)
    return X, mask_arr, y, meta


class MonthSequenceDataset(Dataset):
    def __init__(self, X, mask, state_idx, y):
        self.X = torch.from_numpy(X).float()
        self.mask = torch.from_numpy(mask)
        self.state_idx = torch.from_numpy(state_idx).long()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.mask[i], self.state_idx[i], self.y[i]


def make_loaders(batch_size=32, test_size=0.3, random_state=42, co2_year=2023):
    df = build_labeled_frame(co2_year=co2_year)
    X, pad_mask, y, meta = build_sequence_tensors(df)
    state_idx = meta["state_idx"].values.astype(np.int64)

    X_tr, X_te, m_tr, m_te, s_tr, s_te, y_tr, y_te = train_test_split(
        X, pad_mask, state_idx, y,
        test_size=test_size, random_state=random_state, stratify=y,
    )
    n_states = int(meta["state_idx"].max() + 1)
    info = {
        "n_states": n_states,
        "n_features": N_FEATURES,
        "seq_len": SEQ_LEN,
        "frame": df,
        "meta": meta,
    }
    train_loader = DataLoader(MonthSequenceDataset(X_tr, m_tr, s_tr, y_tr), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MonthSequenceDataset(X_te, m_te, s_te, y_te), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, info


def tabular_last_timestep(X: np.ndarray) -> np.ndarray:
    return X[:, -1, :].copy()


__all__ = [
    "build_labeled_frame",
    "build_sequence_tensors",
    "load_power_state_month_table",
    "load_state_tetce",
    "make_loaders",
    "monthly_top_green_states",
    "tabular_last_timestep",
    "MonthSequenceDataset",
]