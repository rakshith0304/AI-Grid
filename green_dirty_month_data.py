"""Green vs dirty month sequences from project CSVs (inputs are read-only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, cast

import numpy as np
import pandas as pd
from scipy import stats
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

# Repo root: .../DAT 490 - Data Science Capstone (parent of AI-Grid/)
_CAPSTONE_ROOT = Path(__file__).resolve().parent.parent.parent
RESEARCH_DATA = _CAPSTONE_ROOT / "research_data"

# Monthly state / sector / fuel generation (EIA); same column layout as legacy `data_final` extract
_POWER_MONTHLY = RESEARCH_DATA / "02_grid_operations/electric_power_operations_monthly_all_states.csv"
_POWER_2010_2024 = RESEARCH_DATA / "02_grid_operations/electric_power_operations_2010_2024.csv"
POWER_CSV = _POWER_MONTHLY if _POWER_MONTHLY.is_file() else _POWER_2010_2024
SEDS_CSV = RESEARCH_DATA / "08_seds_state_energy/complete_seds_all_energy_1960_2023.csv"

# ISO & RTO fuel-mix extracts (hourly or sub-hourly), see ``research_data/DATA_MANIFEST.csv``
GRID_STRESS_DIR = RESEARCH_DATA / "07_grid_stress"
CAISO_FUEL_MIX_CSV = GRID_STRESS_DIR / "caiso_fuel_mix.csv"
ERCO_FUEL_MIX_HOURLY_CSV = GRID_STRESS_DIR / "erco_fuel_mix_hourly.csv"
ISONE_FUEL_MIX_CSV = GRID_STRESS_DIR / "isone_fuel_mix.csv"
NYISO_FUEL_MIX_CSV = GRID_STRESS_DIR / "nyiso_fuel_mix.csv"
PJM_FUEL_MIX_HOURLY_CSV = GRID_STRESS_DIR / "pjm_fuel_mix_hourly.csv"
RETAIL_PRICE_CSV = RESEARCH_DATA / "03_electricity_prices/retail_sales_price_revenue_monthly_all_states.csv"

SEQ_LEN = 6
# fos_share, nuc_share, month_sin, month_cos, log1p(all_gen), tetce_z — no ren_share (avoids label leakage)
N_FEATURES = 6

# EIA ``fueltypeid`` values when the CSV uses per-technology rows instead of aggregated FOS / REN / NUC.
_REN_DETAIL = frozenset(
    {"REN", "HYC", "SUN", "WND", "GEO", "WAS", "DFO", "MSW", "BIO", "MLG"}
)
_FOS_DETAIL = frozenset({"FOS", "COL", "NG", "OIL", "GAS", "PET", "PC", "SC", "LPG"})
_POWER_USECOLS: tuple[str, ...] = ("period", "location", "sectorid", "fueltypeid", "generation")


def _state_codes_from_power(df: pd.DataFrame) -> list[str]:
    return sorted(
        s for s in df["location"].unique()
        if len(str(s)) == 2 and str(s).isalpha() and s != "US"
    )


def load_state_tetce(year: int = 2023) -> pd.DataFrame:
    sed = pd.read_csv(
        SEDS_CSV,
        usecols=cast(Any, ["MSN", "StateCode", "Year", "Data"]),
    )
    sed = sed[(sed["MSN"] == "TETCE") & (sed["Year"] == year)]
    return sed.rename(columns={"Data": "tetce_mmt", "StateCode": "state"})[["state", "tetce_mmt"]]


def _sum_fuel_cols(piv: pd.DataFrame, codes: frozenset[str]) -> pd.Series:
    cols = [c for c in piv.columns if c in codes]
    if not cols:
        return pd.Series(0.0, index=piv.index, dtype="float64")
    return piv[cols].fillna(0.0).sum(axis=1)


def load_power_state_month_table() -> pd.DataFrame:
    if not POWER_CSV.is_file():
        raise FileNotFoundError(
            f"Power generation CSV not found. Tried: {POWER_CSV}. "
            "Place `electric_power_operations_*.csv` under research_data/02_grid_operations/."
        )
    df = pd.read_csv(
        POWER_CSV,
        usecols=cast(Any, list(_POWER_USECOLS)),
    )
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce").fillna(0.0)
    df["sectorid"] = pd.to_numeric(df["sectorid"], errors="coerce")
    states = _state_codes_from_power(df)
    # Keep all fuel rows (aggregated FOS/REN/NUC *or* per-technology codes such as COL, NG, WND).
    df = df[(df["location"].isin(states)) & (df["sectorid"] == 1)].copy()
    piv = df.pivot_table(index=["period", "location"], columns="fueltypeid", values="generation", aggfunc="sum")
    if "ALL" not in piv.columns:
        piv["ALL"] = 0.0
    piv["ALL"] = piv["ALL"].fillna(0.0)

    if "REN" in piv.columns:
        ren = piv["REN"].fillna(0.0)
    else:
        ren = _sum_fuel_cols(piv, _REN_DETAIL)

    if "FOS" in piv.columns:
        fos = piv["FOS"].fillna(0.0)
    else:
        fos = _sum_fuel_cols(piv, _FOS_DETAIL)

    if "NUC" in piv.columns:
        nuc = piv["NUC"].fillna(0.0)
    else:
        nuc = pd.Series(0.0, index=piv.index, dtype="float64")

    piv["REN"] = ren
    piv["FOS"] = fos
    piv["NUC"] = nuc
    piv = piv[piv["ALL"] > 0].copy()
    piv["ren_share"] = (piv["REN"] / piv["ALL"]).clip(0.0, 1.0)
    piv["fos_share"] = (piv["FOS"] / piv["ALL"]).clip(0.0, 1.0)
    piv["nuc_share"] = (piv["NUC"] / piv["ALL"]).clip(-0.01, 1.0)
    piv = piv.reset_index()
    piv["period"] = pd.to_datetime(piv["period"])
    piv["month"] = piv["period"].dt.month
    piv["year"] = piv["period"].dt.year
    return piv


def load_retail_price_state_month(sector: str = "ALL") -> pd.DataFrame:
    """
    Monthly average retail price (cents per kWh) by state (EIA retail sales).

    ``sector`` is one of ``ALL``, ``RES``, ``COM``, ``IND``, ``TRA``, ``OTH``.
    """
    df = pd.read_csv(
        RETAIL_PRICE_CSV,
        usecols=cast(Any, ["period", "stateid", "sectorid", "price"]),
    )
    df = df[df["sectorid"] == sector].copy()
    df["period"] = pd.to_datetime(df["period"])
    df["price_ckwh"] = pd.to_numeric(df["price"], errors="coerce")
    return df[["period", "stateid", "price_ckwh"]].rename(columns={"stateid": "location"})


def renewable_share_vs_retail_price(price_sector: str = "ALL") -> pd.DataFrame:
    """State–month table: renewable generation share and retail price (inner-join on date and state)."""
    gen = load_power_state_month_table()[["period", "location", "ren_share"]]
    px = load_retail_price_state_month(sector=price_sector)
    out = gen.merge(px, on=["period", "location"], how="inner")
    return out.dropna(subset=["ren_share", "price_ckwh"])


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
    groups = meta["location"].astype(str).values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(np.arange(len(y), dtype=np.int64), y, groups=groups))

    X_tr, X_te = X[train_idx], X[test_idx]
    m_tr, m_te = pad_mask[train_idx], pad_mask[test_idx]
    s_tr, s_te = state_idx[train_idx], state_idx[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    n_states = int(state_idx.max() + 1)
    pos = float(y.mean())
    info = {
        "n_states": n_states,
        "n_features": N_FEATURES,
        "seq_len": SEQ_LEN,
        "frame": df,
        "meta": meta,
        "class_balance": {"P_green": pos, "P_dirty": 1.0 - pos},
        "y_train": y_tr,
        "y_test": y_te,
        "X_train": X_tr,
        "X_test": X_te,
    }
    train_loader = DataLoader(
        MonthSequenceDataset(X_tr, m_tr, s_tr, y_tr), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        MonthSequenceDataset(X_te, m_te, s_te, y_te), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, info


def tabular_last_timestep(X: np.ndarray) -> np.ndarray:
    return X[:, -1, :].copy()


_STATE_NAMES: dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "DC": "District of Columbia",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}


def _linear_trend_prediction_interval(
    t: np.ndarray,
    y_pct: np.ndarray,
    t0: float,
) -> tuple[float, float, float] | None:
    """
    OLS of y on t (time); return (point forecast %, lower 10%, upper 90%) for 80% PI at t0.
    y_pct: renewable share in percent (0–100).
    """
    t = np.asarray(t, dtype=np.float64)
    y_pct = np.asarray(y_pct, dtype=np.float64)
    n = int(t.size)
    if n < 4:
        return None
    t_bar = float(t.mean())
    sxx = float(((t - t_bar) ** 2).sum())
    if sxx <= 1e-12:
        return None
    b = float(((t - t_bar) * (y_pct - y_pct.mean())).sum() / sxx)
    a = float(y_pct.mean() - b * t_bar)
    y_hat = float(a + b * t0)
    resid = y_pct - (a + b * t)
    mse = float((resid**2).sum() / max(n - 2, 1))
    se_pred = float(np.sqrt(mse * (1.0 + 1.0 / n + (t0 - t_bar) ** 2 / sxx)))
    df = max(n - 2, 1)
    t_lo = float(stats.t.ppf(0.10, df))
    t_hi = float(stats.t.ppf(0.90, df))
    lo = y_hat + t_lo * se_pred
    hi = y_hat + t_hi * se_pred
    return (y_hat, lo, hi)


def projected_renewable_usage_top_states(
    target: str = "2028-12-01",
    top_n: int = 7,
    min_months: int = 24,
    exclude_locations: tuple[str, ...] = ("DC", "ME", "NH"),
    replace_in_top: tuple[tuple[str, str], ...] = (("MT", "CA"),),
) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    """
    Linear trend on monthly state renewable share (% generation) from EIA power data;
    extrapolate to ``target`` and build an 80% prediction interval per state.

    By default ``exclude_locations`` omits DC, Maine, and New Hampshire from ranking.
    ``replace_in_top`` swaps out a state in the initial top-``top_n`` cut for another
    (e.g. replace Montana with California) using the replacement’s forecast from the same model.

    Returns
    -------
    table : DataFrame
        Top ``top_n`` states by projected renewable share, columns:
        ``state``, ``projected_pct``, ``lower_pct``, ``upper_pct``, ``band_width``,
        ``starred`` (wide-uncertainty flag).
    national : Series
        ``projected_pct``, ``lower_pct``, ``upper_pct`` for generation-weighted
        national mix (same trend on aggregate monthly REN/ALL).
    meta : dict
        ``data_start`` and ``data_end`` strings for footnotes.
    """
    df = load_power_state_month_table()
    origin = pd.Timestamp(df["period"].min())
    end_data = pd.Timestamp(df["period"].max())
    t_scale = 365.25  # years since origin
    t_target = (pd.Timestamp(target) - origin).days / t_scale

    def series_forecast(periods: pd.Series, y_share: pd.Series) -> tuple[float, float, float] | None:
        g = pd.DataFrame({"period": periods, "y": y_share}).dropna()
        g = g.groupby("period", as_index=False)["y"].mean().sort_values("period")
        if len(g) < min_months:
            return None
        tt = (g["period"] - origin).dt.days.values.astype(float) / t_scale
        y_pct = (g["y"].values * 100.0).astype(float)
        out = _linear_trend_prediction_interval(tt, y_pct, t_target)
        if out is None:
            return None
        y_hat, lo, hi = out
        y_hat = float(np.clip(y_hat, 0.0, 100.0))
        lo = float(np.clip(lo, 0.0, 100.0))
        hi = float(np.clip(hi, 0.0, 100.0))
        if hi < lo:
            lo, hi = hi, lo
        return (y_hat, lo, hi)

    rows: list[dict[str, Any]] = []
    excluded = {str(x).upper().strip() for x in exclude_locations}
    for loc, g in df.groupby("location"):
        if str(loc).upper().strip() in excluded:
            continue
        fc = series_forecast(g["period"], g["ren_share"])
        if fc is None:
            continue
        y_hat, lo, hi = fc
        name = _STATE_NAMES.get(str(loc), str(loc))
        rows.append(
            {
                "state": str(loc),
                "state_name": name,
                "projected_pct": y_hat,
                "lower_pct": lo,
                "upper_pct": hi,
                "band_width": hi - lo,
            }
        )

    all_states = pd.DataFrame(rows)
    if all_states.empty:
        raise ValueError("No state met min_months for trend projection.")
    all_states = all_states.sort_values("projected_pct", ascending=False).reset_index(drop=True)
    top = all_states.head(top_n).copy()
    for old_code, new_code in replace_in_top:
        o = str(old_code).upper().strip()
        n = str(new_code).upper().strip()
        if o not in top["state"].str.upper().values:
            continue
        if n in top["state"].str.upper().values:
            top = top[top["state"].str.upper() != o].reset_index(drop=True)
            continue
        new_row = all_states[all_states["state"].str.upper() == n]
        if new_row.empty:
            raise ValueError(
                f"Replacement state {new_code!r} has no projection (missing data or min_months)."
            )
        top = top[top["state"].str.upper() != o].reset_index(drop=True)
        top = pd.concat([top, new_row.iloc[:1]], ignore_index=True)
        top = top.sort_values("projected_pct", ascending=False).reset_index(drop=True)
    bw = top["band_width"].values.astype(float)
    order = np.argsort(-bw)
    star = np.zeros(len(top), dtype=bool)
    for k in range(min(2, len(order))):
        star[order[k]] = True
    top["starred"] = star

    # National: aggregate generation by month, then trend on share
    nat = (
        df.groupby("period", as_index=False)
        .agg(ren=("REN", "sum"), allg=("ALL", "sum"))
    )
    nat["share"] = (nat["ren"] / nat["allg"].replace(0, np.nan)).clip(0.0, 1.0)
    nfc = series_forecast(nat["period"], nat["share"])
    if nfc is None:
        national = pd.Series(
            {"projected_pct": np.nan, "lower_pct": np.nan, "upper_pct": np.nan, "band_width": np.nan}
        )
    else:
        y_hat, lo, hi = nfc
        national = pd.Series(
            {
                "projected_pct": y_hat,
                "lower_pct": lo,
                "upper_pct": hi,
                "band_width": hi - lo,
            }
        )

    meta = {"data_start": origin.strftime("%Y-%m"), "data_end": end_data.strftime("%Y-%m")}
    return top, national, meta


__all__ = [
    "build_labeled_frame",
    "build_sequence_tensors",
    "CAISO_FUEL_MIX_CSV",
    "ERCO_FUEL_MIX_HOURLY_CSV",
    "GRID_STRESS_DIR",
    "ISONE_FUEL_MIX_CSV",
    "load_power_state_month_table",
    "load_state_tetce",
    "make_loaders",
    "monthly_top_green_states",
    "NYISO_FUEL_MIX_CSV",
    "PJM_FUEL_MIX_HOURLY_CSV",
    "POWER_CSV",
    "RESEARCH_DATA",
    "renewable_share_vs_retail_price",
    "RETAIL_PRICE_CSV",
    "load_retail_price_state_month",
    "projected_renewable_usage_top_states",
    "SEDS_CSV",
    "tabular_last_timestep",
    "MonthSequenceDataset",
]