"""Deterministic repairs established in Stage 1; never overwrite raw files."""
import numpy as np
import pandas as pd
from .common import ROOT, config
from .data_checks import normalize_language, verify_hashes


def clean_training(frame, settings):
    result = frame.drop_duplicates().copy(deep=True)
    result = result.drop(columns=[settings["leaking_column"]], errors="ignore")
    result["language"] = normalize_language(result["language"], settings["language_aliases"])
    result.loc[result.queue_wait_sec.eq(settings["queue_sentinel"]), "queue_wait_sec"] = np.nan
    dates = pd.to_datetime(result["date"], errors="raise")
    stuck = dates.between(settings["stuck_start"], settings["stuck_end"])
    result.loc[stuck, "issue_count"] = np.nan
    # Stage 1 established the isolated 10x cluster; never use phone-bill as input.
    typo = result.handle_min.ge(settings["target_typo_minimum"])
    result.loc[typo, "handle_min"] = result.loc[typo, "handle_min"] / 10
    return result


def load_clean_training():
    cfg = config()
    verify_hashes(ROOT, ROOT / cfg["paths"]["raw"], ROOT / cfg["paths"]["hash_manifest"])
    raw = pd.read_csv(ROOT / cfg["paths"]["raw"] / "train.csv")
    clean = clean_training(raw, cfg["cleaning"])
    destination = ROOT / "data/processed/train_clean.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(destination, index=False)
    return clean
