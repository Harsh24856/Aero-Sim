"""Reader for the released NASA C-MAPSS text files.

STRICTLY READ-ONLY. The files under CMAPSSData/ are the reference against which
our generator is judged; nothing in this project writes to them. The loader
asserts the published unit counts on every read, so a truncated or substituted
file fails immediately instead of quietly shifting the replication target.

FILE FORMAT (CMAPSSData/readme.txt)
    26 whitespace-separated columns per row, one row per operational cycle:
    unit, cycle, 3 operating settings, 21 sensor measurements.
    Rows are grouped by unit and ordered by cycle.

    RUL_FDxxx.txt carries one integer per test unit, in unit order: the true
    remaining cycles after that unit's last recorded cycle.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config as C

def _discover_data_dir() -> str | None:
    """Walk up from this file looking for a CMAPSSData/ directory.

    The data sits outside the package and its depth differs between a normal
    checkout and a git worktree (which nests under .claude/worktrees/<name>), so
    a fixed number of dirname() calls is wrong in one of the two. Searching
    upward works in both.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    while True:
        cand = os.path.join(here, "CMAPSSData")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def _resolve(data_dir: str | None) -> str:
    d = data_dir or os.environ.get("CMAPSS_DATA_DIR") or _discover_data_dir()
    if not d or not os.path.isdir(d):
        raise FileNotFoundError(
            "C-MAPSS reference data not found. Expected a CMAPSSData/ directory in "
            "this repository or an ancestor. Set CMAPSS_DATA_DIR or pass data_dir=."
        )
    return d


def load_split(subset: str, split: str, data_dir: str | None = None) -> pd.DataFrame:
    """Load one subset/split as a DataFrame with named columns.

    Returns columns: unit, cycle, op_setting_1..3, then the 21 Table-2 sensor
    names (T2, T24, ... W32) rather than s1..s21, so downstream code reads as
    physics instead of indices.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if subset not in C.SUBSETS:
        raise ValueError(f"unknown subset {subset!r}")

    path = os.path.join(_resolve(data_dir), f"{split}_{subset}.txt")
    # sep='\s+' handles the trailing whitespace the released files carry, which
    # otherwise yields two spurious all-NaN columns.
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    if df.shape[1] != len(C.ALL_COLS):
        raise ValueError(
            f"{path}: expected {len(C.ALL_COLS)} columns, found {df.shape[1]}. "
            f"The file may be modified or of a different C-MAPSS release."
        )
    df.columns = C.ALL_COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    expected = C.SUBSETS[subset][f"{split}_units"]
    found = df["unit"].nunique()
    if found != expected:
        raise ValueError(
            f"{path}: expected {expected} units per the published readme, found {found}. "
            f"Reference data appears altered - replication target is not trustworthy."
        )
    return df


def load_test_rul(subset: str, data_dir: str | None = None) -> np.ndarray:
    """True remaining cycles for each test unit, in unit order (1-indexed units)."""
    path = os.path.join(_resolve(data_dir), f"RUL_{subset}.txt")
    rul = pd.read_csv(path, sep=r"\s+", header=None, engine="python").iloc[:, 0].to_numpy(int)
    expected = C.SUBSETS[subset]["test_units"]
    if rul.size != expected:
        raise ValueError(f"{path}: expected {expected} RUL values, found {rul.size}")
    return rul


def trajectory_lengths(df: pd.DataFrame) -> np.ndarray:
    """Cycles per unit, ordered by unit id.

    This is the single most diagnostic statistic in the whole replication: the
    generator never sets trajectory length, it falls out of the failure
    criterion H = 0. Matching this distribution is evidence the damage model and
    health index are right, not evidence of tuning.
    """
    return df.groupby("unit")["cycle"].max().sort_index().to_numpy()


def add_train_rul(df: pd.DataFrame, cap: int | None = C.RUL_CAP) -> pd.DataFrame:
    """Attach RUL labels to a TRAIN split, where every unit runs to failure.

    A training unit's last cycle is failure, so RUL = max_cycle - cycle. The cap
    applies the standard piecewise-linear convention (config.RUL_CAP); pass
    cap=None for the raw linear label.
    """
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = last - out["cycle"]
    if cap is not None:
        out["RUL"] = out["RUL"].clip(upper=cap)
    return out


def add_test_rul(df: pd.DataFrame, true_rul: np.ndarray, cap: int | None = C.RUL_CAP) -> pd.DataFrame:
    """Attach RUL labels to a TEST split, whose units stop before failure.

    Each unit's trajectory is truncated, so the true RUL at its final cycle is
    given by RUL_FDxxx.txt rather than being derivable from the data. RUL at any
    earlier cycle is that value plus the cycles still to come.
    """
    out = df.copy()
    units = np.sort(out["unit"].unique())
    if units.size != true_rul.size:
        raise ValueError(f"{units.size} units but {true_rul.size} RUL values")
    tail = pd.Series(true_rul, index=units, name="tail_rul")
    last = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = out["unit"].map(tail).to_numpy() + (last - out["cycle"]).to_numpy()
    if cap is not None:
        out["RUL"] = out["RUL"].clip(upper=cap)
    return out


def constant_sensors(df: pd.DataFrame, tol: float = 1e-9) -> list[str]:
    """Sensors carrying no information in this subset.

    In the single-condition subsets several channels are flat to machine
    precision and are conventionally dropped. Reported rather than hardcoded so
    the choice is derived from the data, and so the generator can be checked for
    reproducing the same flatness.
    """
    return [c for c in C.SENSOR_COLS if df[c].std(ddof=0) <= tol]
