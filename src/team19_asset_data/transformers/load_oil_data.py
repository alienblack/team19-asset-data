"""Cleaning for the Mendeley power-transformer oil-analysis dataset.

This module only cleans. It runs no experiment, fits no model and makes no
modelling decision; those belong in ``notebooks/transformers_imputation.ipynb``.

Source
------
https://data.mendeley.com/datasets/98f4z3f8tx/2
Public research data describing dissolved-gas and oil-quality tests performed on
real power (substation) transformers over multiple years of service.

NOT SA Power Networks data, and not a pole-mounted distribution asset register.
The raw download is not committed; the cleaned output lives in
``data/synthetic/transformers/power_transformers_oil.csv`` and is what the
notebook reads via :func:`load_clean_dataset`.
"""

from pathlib import Path

import pandas as pd

RAW_FILE = "data/raw/Large Paramater Dataset/Base_de_Datos_Eng_5_2_2026_V1.csv"
OUT_FILE = "data/synthetic/transformers/power_transformers_oil.csv"

#: Columns that stay textual; everything else is coerced to a number.
TEXT_FIELDS = [
    "Transformer", "Visual Inspection", "Corrosive Sulfur Test",
    "Inhibitor Content ()", "Diagnosis by Basic Gas Ratio",
    "Diagnosis by Dornenburg", "Diagnosis by Rogers", "Diagnosis by Duval",
]
#: Furan columns use "ND,<10" for a below-detection-limit reading.
FURAN_FIELDS = [
    "Hydroxymethyl Furan (ppb)", "Furfuryl Alcohol (ppb)", "Furfural (ppb)",
    "Acetyl Furan (ppb)", "Methyl Furfural (ppb)",
]
#: Columns emptier than this carry no usable signal and are dropped.
DROP_ABOVE_MISSING = 0.7

#: ASTM D1500 oil colour runs 0.5-8.0; readings outside it are recording errors.
COLOUR_RANGE = (0.5, 8.0)
COLOUR_BINS = [0, 0.5, 1.0, 1.75, 2.75, 8.0]
COLOUR_LABELS = ["<=0.5", "0.75-1.0", "1.25-1.75", "2.0-2.75", ">=3.0"]


def _to_number(s: pd.Series) -> pd.Series:
    """Coerce a European-formatted column, keeping the first of a dual rating."""
    t = s.astype("string").str.strip().str.split("/").str[0]
    return pd.to_numeric(t.str.replace(",", ".", regex=False), errors="coerce")


def _to_furan(s: pd.Series) -> pd.Series:
    """Below-detection-limit ("ND,<10") becomes 0.0; a blank cell stays NaN."""
    t = s.astype("string").str.strip()
    below = t.str.upper().str.startswith("ND", na=False)
    return pd.to_numeric(t.str.replace(",", ".", regex=False), errors="coerce").mask(below, 0.0)


def build_clean_dataset(repo_root: Path | str = ".") -> pd.DataFrame:
    """Read the raw Mendeley CSV and return it as a tidy DataFrame.

    Parameters
    ----------
    repo_root
        Repository root, so the function works from a notebook in ``notebooks/``
        as well as from the project root.

    Returns
    -------
    pandas.DataFrame
        One row per oil sample: 933 measurements across 105 transformers.
        Columns keep their original names. ``Transformer`` is the unit
        identifier; ``Tp`` / ``Ts`` / ``Power (MVA)`` / ``Year`` are nameplate
        attributes; the remainder are oil-test results and legacy diagnoses.
        A derived ordinal ``colour_band`` is added.

    Cleaning rules applied
    ----------------------
    1. Read as semicolon-separated, latin-1 encoded; drop the trailing unnamed
       column and strip whitespace from headers.
    2. Keep only the leading block. The raw file stacks two datasets: the first
       933 rows carry transformer metadata, the remainder are gas-only records
       with no identifier and cannot be joined to an asset.
    3. Convert European decimal commas to points, and keep the first value of a
       dual rating such as "10,0/14,0".
    4. Map furan "ND,<10" readings to 0.0 - a below-detection-limit result is a
       measurement, not a missing value, and must not be confused with one.
    5. Normalise whitespace in the text columns.
    6. Bin ASTM D1500 oil colour into ordinal bands, leaving out-of-range
       readings as NaN. No row is dropped for a missing colour.
    7. Drop columns more than ``DROP_ABOVE_MISSING`` empty. Rows are never
       dropped for missingness: complete-case deletion would leave 2 of 933
       rows and would bias the sample towards well-documented transformers.
    """
    root = Path(repo_root)
    df = pd.read_csv(root / RAW_FILE, sep=";", encoding="latin-1")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.strip()

    df = df.loc[: df["Transformer"].isna().idxmax() - 1].copy()

    for col in FURAN_FIELDS:
        df[col] = _to_furan(df[col])
    for col in df.columns:
        if col not in TEXT_FIELDS and col not in FURAN_FIELDS:
            df[col] = _to_number(df[col])
    for col in TEXT_FIELDS:
        df[col] = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)

    valid = df["Color"].where(df["Color"].between(*COLOUR_RANGE))
    df["colour_band"] = pd.cut(valid, COLOUR_BINS, labels=COLOUR_LABELS).astype("string")

    df = df.drop(columns=df.columns[df.isna().mean() > DROP_ABOVE_MISSING])
    return df.reset_index(drop=True)


def load_clean_dataset(repo_root: Path | str = ".") -> pd.DataFrame:
    """Read the committed cleaned CSV (``OUT_FILE``) with the same dtypes that
    :func:`build_clean_dataset` produces, so the notebook does not need the raw
    download.
    """
    root = Path(repo_root)
    text_cols = {c: "string" for c in TEXT_FIELDS + ["colour_band"]}
    df = pd.read_csv(root / OUT_FILE, dtype=text_cols)
    return df


def main(repo_root: Path | str = ".") -> None:
    """Write the cleaned dataset to ``OUT_FILE``."""
    root = Path(repo_root)
    df = build_clean_dataset(root)
    out = root / OUT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  shape={df.shape}  transformers={df['Transformer'].nunique()}")


if __name__ == "__main__":
    main()
