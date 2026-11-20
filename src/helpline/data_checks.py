"""Small, read-only checks used by the Stage 1 exploration command."""

import hashlib
import json

import pandas as pd


def verify_hashes(root, raw_directory, manifest_path):
    """Verify required raw files and any retained extracted originals."""
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, digest in expected.items():
        paths = [raw_directory / name]
        original = root / "data" / name
        if original.exists():
            paths.append(original)
        for path in paths:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"Original data hash mismatch: {path}")
    return expected


def stuck_periods(frame, column, minimum_days=7):
    """Find consecutive full days on which every record has the same value.

    Missing calendar days and mixed-value days break a run. This avoids
    extending a sensor fault into an adjacent day with one matching record.
    """
    dated = frame.assign(_date=pd.to_datetime(frame["date"], errors="raise"))
    daily = dated.groupby("_date")[column].agg(["size", "count", "nunique", "first"])
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar)
    constant = daily["nunique"].eq(1) & daily["count"].eq(daily["size"])
    breaks = ~constant | ~constant.shift(fill_value=False)
    breaks |= daily["first"].ne(daily["first"].shift())
    periods = []
    for _, run in daily.groupby(breaks.cumsum()):
        if len(run) >= minimum_days and constant.loc[run.index].all():
            periods.append({
                "column": column,
                "value": float(run["first"].iloc[0]),
                "start_date": run.index[0].strftime("%Y-%m-%d"),
                "end_date": run.index[-1].strftime("%Y-%m-%d"),
                "days": len(run),
                "rows": int(run["size"].sum()),
            })
    return periods


def target_outliers(target, minimum_ratio=2.0):
    """Screen for a separated upper cluster; this alone does not prove typos."""
    ordered = target.dropna().sort_values().reset_index(drop=True)
    previous = ordered.shift()
    ratios = ordered / previous.where(previous > 0)
    # Only inspect upper-tail boundaries, not ratios among very small targets.
    ratios = ratios.where(previous >= ordered.quantile(0.75))
    candidates = ratios[ratios >= minimum_ratio]
    if candidates.empty:
        return pd.Series(False, index=target.index), None
    boundary = candidates.idxmax()
    lower, upper = float(previous.iloc[boundary]), float(ordered.iloc[boundary])
    return target.ge(upper), {"lower": lower, "upper": upper, "ratio": upper / lower}


def normalize_language(values, aliases):
    """Diagnostic normalization on a new Series; does not modify raw data."""
    return values.str.strip().str.lower().replace(aliases)
