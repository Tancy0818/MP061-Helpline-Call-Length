"""Run Stage 1 checks; write evidence and only the data portion of findings."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json

import pandas as pd
import yaml

if __package__:
    from .data_checks import normalize_language, stuck_periods, target_outliers, verify_hashes
else:
    from data_checks import normalize_language, stuck_periods, target_outliers, verify_hashes


ROOT = Path(__file__).resolve().parents[2]


def analyse(train, later, holdout, settings):
    target, group = settings["target"], settings["group"]
    features = settings["numeric_features"]
    bill = settings["post_event_column"]
    print("Rows:", {"train": len(train), "later": len(later), "holdout": len(holdout)})
    print("Training missing values:\n", train.isna().sum().to_string())
    print("Numeric summary:\n", train.describe().to_string())

    # Target corruption can hide the otherwise near-perfect leakage correlation.
    typo_mask, gap = target_outliers(train[target], settings["minimum_target_gap_ratio"])
    normal = ~typo_mask
    normal_error = (train.loc[normal, target] - train.loc[normal, bill]).abs()
    corrected_error = (train.loc[typo_mask, target] / 10 - train.loc[typo_mask, bill]).abs()
    typo_confirmed = bool(typo_mask.any() and corrected_error.max() <= normal_error.max())
    if typo_mask.any() and not typo_confirmed:
        raise ValueError("Upper target cluster needs manual investigation; not confirmed as 10x typos.")
    print("\nTARGET TYPOS: upper-tail gap", gap)
    print("Confirmed 10x typo count:", int(typo_mask.sum()))
    print(train.loc[typo_mask, ["record_id", "date", target, bill]].to_string(index=False))
    print("Max phone-bill difference, normal targets:", float(normal_error.max()))
    print("Max difference after diagnostic division by 10:", float(corrected_error.max()))

    absent = sorted(set(train.columns) - set(holdout.columns))
    correlation = float(train.loc[normal, target].corr(train.loc[normal, bill]))
    leak_present = bool(bill in absent and correlation > 0.98)
    print("\nLEAKAGE: columns absent from holdout:", absent)
    print("Phone-bill correlation on non-typo rows:", correlation)
    print("PROBLEM.md says the bill is available later, after the prediction event.")

    print("\nSENTINELS: repeated special values across numeric predictors")
    sentinels = []
    for column in features:
        print(column, "most frequent:", train[column].value_counts().head(5).to_dict())
        for value in settings["sentinel_candidates"]:
            count = int(train[column].eq(value).sum())
            if count:
                sentinels.append({"column": column, "value": value, "count": count})
    print("Candidates:", sentinels)
    # The client explicitly rules out negative waits; don't label arbitrary
    # special numbers in other columns as sentinels without investigation.
    if any(s["column"] != "queue_wait_sec" or s["value"] >= 0 for s in sentinels):
        raise ValueError("Unexpected sentinel candidate requires review.")
    if len(sentinels) > 1:
        raise ValueError("Multiple sentinel candidates require review.")

    print("\nUNIT MIXUP: within-team summaries, excluding impossible wait sentinels")
    scale_candidates = []
    for column in features:
        series = train[column].mask(train[column].isin(settings["sentinel_candidates"]))
        summary = series.groupby(train[group]).agg(["min", "median", "mean", "max"])
        ratio = float(summary["median"].max() / summary["median"].min())
        print(column, "\n", summary.to_string(), "\nLargest/smallest median:", ratio)
        if ratio >= settings["unit_scale_screen_ratio"]:
            scale_candidates.append(column)
    if scale_candidates:
        raise ValueError(f"Review exact unit conversions for: {scale_candidates}")
    print("No group-scale separation: ranges overlap in the documented units.")
    print("This is a dataset-specific conclusion, not a universal unit-error detector.")

    duplicates = int(train.duplicated().sum())
    print("\nDUPLICATES: extra identical rows:", duplicates)
    print("Repeated record IDs:", int(train["record_id"].duplicated().sum()))

    print("\nSTUCK SENSOR: consecutive full-day constant runs")
    periods = []
    for column in train.select_dtypes(include="number").columns:
        periods.extend(stuck_periods(train, column, settings["minimum_stuck_days"]))
    print(json.dumps(periods, indent=2))
    if len(periods) > 1:
        raise ValueError("Multiple long constant runs require review.")
    for period in periods:
        daily = train.groupby("date")[period["column"]].agg(["size", "nunique", "min", "max"])
        start = (pd.Timestamp(period["start_date"]) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        end = (pd.Timestamp(period["end_date"]) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        print("Boundary evidence:\n", daily.loc[start:end].to_string())

    print("\nSPELLING: exact labels (repr exposes spaces)")
    for column in ["language", group]:
        for value, count in train[column].value_counts().items():
            print(column, repr(value), int(count))
        print("Later clean levels:", sorted(later[column].unique()))
    language = normalize_language(train["language"], settings["language_aliases"])
    if set(language) != set(later["language"]):
        raise ValueError("Unresolved language category; review mapping.")
    if set(train[group]) != set(later[group]):
        raise ValueError("Unexpected team categories require review.")
    changed = int(language.ne(train["language"]).sum())
    print("Language labels changed diagnostically:", changed)
    print("Normalized counts:", language.value_counts().to_dict())

    return {
        "leaking_column": {"present": leak_present, "column": bill if leak_present else None},
        "unit_mixup": {"present": False, "column": None, "group_value": None, "factor": None},
        "sentinel_value": {"present": bool(sentinels), **(sentinels[0] if sentinels else
                            {"column": None, "value": None, "count": None})},
        "duplicate_rows": {"present": duplicates > 0, "count": duplicates or None},
        "stuck_sensor": {
            "present": bool(periods),
            **{key: periods[0][key] if periods else None
               for key in ["column", "start_date", "end_date"]},
        },
        "target_typos": {"present": typo_confirmed, "count": int(typo_mask.sum()) or None},
        "spelling_variants": {"present": changed > 0, "column": "language" if changed else None,
                              "true_levels": int(language.nunique()) if changed else None},
    }


def main():
    config = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    paths = config["paths"]
    raw = ROOT / paths["raw"]
    hashes = verify_hashes(ROOT, raw, ROOT / paths["hash_manifest"])
    report = StringIO()
    with redirect_stdout(report):
        print("G247 MP061 - Stage 1 evidence; original rows, no cleaning applied")
        print("Verified SHA-256 fingerprints:", json.dumps(hashes, indent=2))
        issues = analyse(
            pd.read_csv(raw / "train.csv"), pd.read_csv(raw / "later.csv"),
            pd.read_csv(raw / "holdout_features.csv"), config["analysis"],
        )
    verify_hashes(ROOT, raw, ROOT / paths["hash_manifest"])
    # Preserve later-stage values if this analysis is rerun in the future.
    findings_path = ROOT / paths["findings"]
    source = findings_path if findings_path.exists() else ROOT / paths["findings_template"]
    findings = yaml.safe_load(source.read_text(encoding="utf-8"))
    findings["data_issues"] = issues
    report_path = ROOT / paths["exploration_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.getvalue(), encoding="utf-8")
    findings_path.write_text(yaml.safe_dump(findings, sort_keys=False), encoding="utf-8")
    print(report.getvalue())
    print("Wrote", report_path.relative_to(ROOT), "and data_issues in", findings_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
