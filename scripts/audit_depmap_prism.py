#!/usr/bin/env python3
"""
Descriptive data-feasibility audit: DepMap expression x PRISM Repurposing response.

Reads LOCAL files only and makes no network calls. It reports what the supplied
files contain; it carries no expected counts and never aggregates duplicates.

Required inputs
  --expression      DepMap expression matrix (e.g. OmicsExpressionProteinCodingGenesTPMLogp1.csv)
  --model           DepMap Model.csv from the SAME release
  --prism-matrix    PRISM response matrix (either orientation; verified from the identifiers)

Optional inputs
  --prism-compounds   PRISM compound annotation table
  --prism-cell-meta   PRISM cell-line annotation table
  --omics-profiles    DepMap OmicsProfiles.csv (library / strandedness diagnostics)

Policies
  --duplicate-policy fail     (default) report duplicated IDs, write duplicates_report.csv,
                              stop with exit code 3 BEFORE overlap/usable counts are computed.
  --duplicate-policy exclude  explicitly exclude EVERY copy of each ambiguous ID (nothing is
                              kept, averaged or aggregated); each exclusion is written to
                              exclusions.csv and counted in the report.
  --allow-nonnumeric          continue when response entries are neither numeric nor a
                              recognised missing token; they are counted separately and are
                              never treated as usable observations.

Exit codes: 0 success | 2 input/format error | 3 blocked by duplicated identifiers

Example
  python scripts/audit_depmap_prism.py \
      --expression data/raw/OmicsExpressionProteinCodingGenesTPMLogp1.csv \
      --model data/raw/Model.csv \
      --prism-matrix data/raw/<PRISM_matrix>.csv \
      --prism-compounds data/raw/<PRISM_compound_annotation>.csv \
      --outdir results/audit
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- constants
ACH_PATTERN = r"ACH-\d{6}"
ACH_FIND = re.compile(ACH_PATTERN)
GENE_COL_RE = re.compile(r"^.+ \(\d+\)$")  # e.g. "TP53 (7157)"
BRD_BASE_RE = re.compile(r"^(BRD-[A-Z]\d{8})")
BRD_PREFIX_RE = re.compile(r"^BRD:", re.IGNORECASE)

# Tokens recognised as "missing" in response matrices. Anything else that is not a finite or
# infinite number is classed as NONNUMERIC and reported separately.
MISSING_TOKENS = frozenset({"", "na", "nan", "n/a", "null", "none"})
TRUE_TOKENS = frozenset({"yes", "true", "1"})
FALSE_TOKENS = frozenset({"no", "false", "0", "", "nan", "none", "na"})

# Orientation classification (identifier-based; see decide_orientation). These are
# classification rules, not expected data counts.
CELL_AXIS_MIN_ACH_FRACTION = 0.8
OTHER_AXIS_MAX_ACH_FRACTION = 0.05
ID_COLUMN_MIN_ACH_FRACTION = 0.9

QUANTILES = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)

# integer codes for response-entry categories
FINITE, MISSING, NONFINITE, NONNUMERIC = 0, 1, 2, 3
CATEGORY_NAMES = {FINITE: "finite_numeric", MISSING: "missing", NONFINITE: "nonfinite", NONNUMERIC: "nonnumeric"}


class AuditError(Exception):
    """Unrecoverable input/format problem. The message is written for the user."""


class IntegrityBlock(Exception):
    """Duplicated identifiers were found and --duplicate-policy is 'fail'."""


# --------------------------------------------------------------------------- parameters / reporting
@dataclass
class AuditParams:
    expression: Path
    model: Path
    prism_matrix: Path
    outdir: Path = Path("results/audit")
    prism_compounds: Path | None = None
    prism_cell_meta: Path | None = None
    omics_profiles: Path | None = None
    prism_orientation: str = "auto"
    expression_id_col: str | None = None
    expression_default_col: str | None = None
    model_id_col: str = "ModelID"
    compound_id_col: str | None = None
    lineage_col: str | None = None
    duplicate_policy: str = "fail"
    allow_nonnumeric: bool = False
    coverage_fractions: tuple = (0.25, 0.5, 0.75, 0.9, 1.0)
    coverage_thresholds: tuple = ()
    sd_thresholds: tuple = ()
    quiet: bool = False


class Report:
    def __init__(self, quiet: bool = False):
        self.lines: list[str] = []
        self.summary: dict = {"exclusions": {}}
        self.exclusion_rows: list[dict] = []
        self.tables: dict[str, pd.DataFrame] = {}
        self.quiet = quiet

    def __call__(self, msg: str = ""):
        self.lines.append(msg)
        if not self.quiet:
            print(msg)

    def section(self, title: str):
        self(f"\n====== {title} ======")


# --------------------------------------------------------------------------- generic helpers
def jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, (set, tuple)):
        return sorted(o) if isinstance(o, set) else list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_csv_checked(path: Path, label: str, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kw)
    except FileNotFoundError:
        raise AuditError(f"{label}: file not found: {path}")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise AuditError(f"{label}: could not parse {path} as CSV ({type(e).__name__}: {e}).")


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def require_columns(df: pd.DataFrame, required, label: str, hint: str = ""):
    """Raise an informative AuditError if required columns are absent."""
    missing = [c for c in required if c not in df.columns]
    if not missing:
        return
    by_norm = {_norm_name(c): str(c) for c in df.columns}
    suggestions = {}
    for m in missing:
        close = difflib.get_close_matches(_norm_name(m), list(by_norm), n=3, cutoff=0.6)
        suggestions[m] = [by_norm[c] for c in close]
    cols = [str(c) for c in df.columns]
    shown = cols[:40] + (["..."] if len(cols) > 40 else [])
    raise AuditError(
        f"{label}: required column(s) {missing} not found. Available columns: {shown}. "
        f"Closest matches: {suggestions}. {hint}".strip()
    )


def ach_fraction(values) -> float:
    s = pd.Series(list(values), dtype="object").dropna().astype(str).str.strip()
    return float(s.str.fullmatch(ACH_PATTERN).mean()) if len(s) else 0.0


def qdict(x) -> dict:
    x = pd.Series(np.asarray(x, dtype=float)).dropna()
    if x.empty:
        return {}
    return {f"q{q}": float(x.quantile(q)) for q in QUANTILES}


def eta_squared(y: pd.Series, groups: pd.Series):
    d = pd.DataFrame({"y": y.to_numpy(), "g": groups.to_numpy()}).dropna()
    if d["g"].nunique() < 2 or len(d) < 3:
        return None
    ss_tot = ((d["y"] - d["y"].mean()) ** 2).sum()
    if ss_tot == 0:
        return None
    ss_between = sum(len(v) * (v.mean() - d["y"].mean()) ** 2 for _, v in d.groupby("g")["y"])
    return float(ss_between / ss_tot)


def find_ach_column(df, candidate_cols, explicit, label, option):
    if explicit:
        require_columns(df, [explicit], label, f"(requested with {option})")
        frac = ach_fraction(df[explicit])
        if frac < ID_COLUMN_MIN_ACH_FRACTION:
            raise AuditError(f"{label}: column {explicit!r} given via {option} has only {frac:.3f} "
                             f"ACH-style IDs (expected >= {ID_COLUMN_MIN_ACH_FRACTION}).")
        return explicit
    cands = [c for c in candidate_cols if ach_fraction(df[c]) >= ID_COLUMN_MIN_ACH_FRACTION]
    if not cands:
        raise AuditError(f"{label}: no column holds ACH-style model IDs ({ACH_PATTERN}). "
                         f"Columns inspected: {[str(c) for c in candidate_cols][:20]}. Use {option}.")
    if len(cands) == 1:
        return cands[0]
    if "ModelID" in cands:
        return "ModelID"
    raise AuditError(f"{label}: several columns look like model IDs {cands}; choose one with {option}.")


# --------------------------------------------------------------------------- response-entry classification
def classify_strings(frame: pd.DataFrame):
    """Classify every entry of a string DataFrame. Nothing is coerced silently.

    Returns (values, codes, examples):
      values   float array; finite numbers only, NaN elsewhere
      codes    int8 array of FINITE / MISSING / NONFINITE / NONNUMERIC
      examples up to 20 (row_pos, col_pos, token) for NONNUMERIC entries
    """
    n_r, n_c = frame.shape
    values = np.full((n_r, n_c), np.nan)
    codes = np.zeros((n_r, n_c), dtype=np.int8)
    examples: list[tuple[int, int, str]] = []
    for j in range(n_c):
        s = frame.iloc[:, j].astype("string").str.strip()
        low = s.str.lower()
        miss = (s.isna() | low.isin(list(MISSING_TOKENS))).to_numpy(dtype=bool)
        num = pd.to_numeric(s.where(~miss), errors="coerce")
        numv = num.to_numpy(dtype=float, na_value=np.nan)
        nonnum = (~miss) & np.isnan(numv)
        nonfin = (~miss) & np.isinf(numv)
        fin = (~miss) & np.isfinite(numv)
        codes[miss, j] = MISSING
        codes[nonfin, j] = NONFINITE
        codes[nonnum, j] = NONNUMERIC
        values[fin, j] = numv[fin]
        if len(examples) < 20:
            for i in np.flatnonzero(nonnum)[: 20 - len(examples)]:
                examples.append((int(i), j, str(s.iloc[int(i)])))
    return values, codes, examples


def category_counts(codes: np.ndarray) -> dict:
    return {name: int((codes == k).sum()) for k, name in CATEGORY_NAMES.items()} | {"total": int(codes.size)}


# --------------------------------------------------------------------------- compound ID normalisation
@dataclass
class NormalizedId:
    original: str
    column_key: str            # normalised, dose/screen tokens kept (identity of a matrix column)
    match_key: str             # column_key without '::' tokens (used to match annotations)
    base_key: str | None       # BRD-X######## (batch/salt suffix removed) - review tier only
    dose_token: str | None
    screen_token: str | None
    rules: list = field(default_factory=list)


def normalize_compound_id(raw) -> NormalizedId:
    """Apply ordered, recorded normalisation rules. The original string is always preserved.

    Rules (each recorded in NormalizedId.rules when it changes or detects something):
      strip_whitespace          leading/trailing whitespace removed
      strip_BRD_colon_prefix    leading 'BRD:' removed ('BRD:BRD-K..' -> 'BRD-K..')
      uppercase_brd_id          'brd-k..' -> 'BRD-K..'
      detect_dose_screen_tokens '<id>::<dose>::<screen>' split; tokens recorded, removed from match_key
      unexpected_extra_tokens   more than two '::' separated tokens (kept in column_key)
      detected_batch_suffix     ID has '-###-##-#' style suffix (detected only, NOT removed)
      no_BRD_pattern            ID does not start with a BRD-X######## pattern (matched verbatim)
    """
    original = "" if raw is None else str(raw)
    rules: list[str] = []
    s = original
    if s != s.strip():
        s = s.strip()
        rules.append("strip_whitespace")
    if BRD_PREFIX_RE.match(s):
        s = s[4:]
        rules.append("strip_BRD_colon_prefix")
    if s.upper().startswith("BRD-") and s != s.upper():
        s = s.upper()
        rules.append("uppercase_brd_id")
    dose = screen = None
    core = s
    if "::" in s:
        parts = s.split("::")
        core = parts[0]
        dose = parts[1] if len(parts) > 1 else None
        screen = parts[2] if len(parts) > 2 else None
        rules.append("detect_dose_screen_tokens")
        if len(parts) > 3:
            rules.append("unexpected_extra_tokens")
    m = BRD_BASE_RE.match(core)
    base = m.group(1) if m else None
    if base is None:
        rules.append("no_BRD_pattern")
    elif len(core) > len(base):
        rules.append("detected_batch_suffix")
    return NormalizedId(original, s, core, base, dose, screen, rules)


def normalization_table(norms: list[NormalizedId], source: str) -> pd.DataFrame:
    return pd.DataFrame({
        "source": source,
        "original": [n.original for n in norms],
        "column_key": [n.column_key for n in norms],
        "match_key": [n.match_key for n in norms],
        "base_key": [n.base_key for n in norms],
        "dose_token": [n.dose_token for n in norms],
        "screen_token": [n.screen_token for n in norms],
        "rules_applied": [";".join(n.rules) for n in norms],
    })


def report_rule_counts(rep: Report, norms: list[NormalizedId], label: str) -> dict:
    counts = Counter(r for n in norms for r in n.rules)
    rep(f"normalisation rules applied to {label} identifiers (original IDs are preserved in "
        f"compound_id_normalization.csv):")
    if not counts:
        rep("  none: every identifier was already in normalised form")
    for rule, n in sorted(counts.items()):
        rep(f"  {rule}: {n}")
    return dict(counts)


# --------------------------------------------------------------------------- orientation
def decide_orientation(row_labels, col_labels, explicit: str, rep: Report | None = None) -> str:
    """Decide orientation from the identifiers themselves. Raises AuditError if ambiguous."""
    r, c = ach_fraction(row_labels), ach_fraction(col_labels)
    n_r, n_c = len(row_labels), len(col_labels)
    rows_are_cells = r >= CELL_AXIS_MIN_ACH_FRACTION and c <= OTHER_AXIS_MAX_ACH_FRACTION
    cols_are_cells = c >= CELL_AXIS_MIN_ACH_FRACTION and r <= OTHER_AXIS_MAX_ACH_FRACTION
    evidence = (f"ACH-ID fraction: row labels={r:.3f} (n={n_r}), column labels={c:.3f} (n={n_c}); "
                f"required: cell axis >= {CELL_AXIS_MIN_ACH_FRACTION}, other axis <= {OTHER_AXIS_MAX_ACH_FRACTION}")
    if rows_are_cells == cols_are_cells:  # both or neither -> ambiguous
        if explicit == "auto":
            raise AuditError(
                f"PRISM matrix orientation is ambiguous. {evidence}. Dimensions are not used as "
                f"evidence because both orientations are plausible. Inspect the file; if you have "
                f"verified it manually, pass --prism-orientation cells_by_compounds|compounds_by_cells.")
        if rep:
            rep(f"WARNING: identifier evidence is ambiguous ({evidence}); using user-declared "
                f"orientation {explicit!r}.")
        return explicit
    detected = "cells_by_compounds" if rows_are_cells else "compounds_by_cells"
    if explicit not in ("auto", detected):
        raise AuditError(f"--prism-orientation {explicit} contradicts the identifiers, which indicate "
                         f"{detected}. {evidence}.")
    return detected


# --------------------------------------------------------------------------- PRISM matrix
@dataclass
class PrismMatrix:
    values: np.ndarray            # cells x compounds; finite numbers, NaN elsewhere
    codes: np.ndarray             # entry categories, same shape
    cell_ids: list                # whitespace-stripped IDs
    compound_original: list       # labels exactly as in the file
    compound_norm: list           # NormalizedId, parallel to columns
    orientation: str = ""
    duplicates: list = field(default_factory=list)

    def drop(self, cell_ids_to_drop: set, column_keys_to_drop: set):
        keep_r = np.array([c not in cell_ids_to_drop for c in self.cell_ids], dtype=bool)
        keep_c = np.array([n.column_key not in column_keys_to_drop for n in self.compound_norm], dtype=bool)
        self.values, self.codes = self.values[keep_r][:, keep_c], self.codes[keep_r][:, keep_c]
        self.cell_ids = [c for c, k in zip(self.cell_ids, keep_r) if k]
        self.compound_original = [c for c, k in zip(self.compound_original, keep_c) if k]
        self.compound_norm = [c for c, k in zip(self.compound_norm, keep_c) if k]


def load_prism_matrix(p: AuditParams, rep: Report) -> PrismMatrix:
    rep.section("3. PRISM response matrix")
    raw = read_csv_checked(p.prism_matrix, "PRISM matrix", header=None, dtype=str,
                           keep_default_na=False, na_filter=False)
    if raw.shape[0] < 2 or raw.shape[1] < 2:
        raise AuditError(f"PRISM matrix: file has shape {raw.shape}; need a header row, a label column and data.")
    corner = raw.iat[0, 0]
    col_labels_raw = raw.iloc[0, 1:].tolist()
    row_labels_raw = raw.iloc[1:, 0].tolist()
    col_labels = [x.strip() for x in col_labels_raw]
    row_labels = [x.strip() for x in row_labels_raw]
    rep(f"raw table: {raw.shape[0] - 1} data rows x {raw.shape[1] - 1} data columns | corner cell: {corner!r}")
    rep(f"ACH-ID fraction: row labels={ach_fraction(row_labels):.3f}, column labels={ach_fraction(col_labels):.3f}")

    orientation = decide_orientation(row_labels, col_labels, p.prism_orientation, rep)
    rep(f"orientation: {orientation} (verified from identifiers; requested: {p.prism_orientation})")

    values, codes, examples = classify_strings(raw.iloc[1:, 1:])
    if orientation == "compounds_by_cells":
        values, codes = values.T, codes.T
        cell_ids, compound_original = col_labels, row_labels_raw
    else:
        cell_ids, compound_original = row_labels, col_labels_raw
    compound_norm = [normalize_compound_id(x) for x in compound_original]
    rep(f"cell lines x compounds: {len(cell_ids)} x {len(compound_original)}")
    if len(cell_ids) != values.shape[0] or len(compound_original) != values.shape[1]:
        raise AuditError("PRISM matrix: label counts do not match matrix dimensions after orientation.")

    cc = category_counts(codes)
    rep("response-entry categories (whole file, before any restriction):")
    for k in ("finite_numeric", "missing", "nonfinite", "nonnumeric"):
        rep(f"  {k}: {cc[k]:,} ({cc[k] / cc['total']:.4%})")
    rep(f"  total: {cc['total']:,}")
    if cc["nonnumeric"]:
        ex = [(("cell" if orientation == "cells_by_compounds" else "compound") + f"_pos={i}",
               ("compound" if orientation == "cells_by_compounds" else "cell") + f"_pos={j}", tok)
              for i, j, tok in examples[:10]]
        rep(f"  nonnumeric examples (first 10; positions are in the file's own row/column order): {ex}")
        if not p.allow_nonnumeric:
            raise AuditError(f"PRISM matrix: {cc['nonnumeric']} entries are neither numeric nor a recognised "
                             f"missing token {sorted(MISSING_TOKENS - {''})}. Examples: {ex}. Inspect the file, or "
                             f"re-run with --allow-nonnumeric to count them separately (they are never usable).")
    fin = values[np.isfinite(values)]
    rep(f"finite-value quantiles: {json.dumps({k: round(v, 4) for k, v in qdict(fin).items()})}")

    tokens = [n.dose_token for n in compound_norm if n.dose_token is not None]
    rep(f"compound labels with dose/screen tokens: {len(tokens)}; distinct dose tokens: "
        f"{sorted(set(tokens))[:10] if tokens else 'none detected'}")
    nrm = report_rule_counts(rep, compound_norm, "PRISM matrix compound")
    keys = Counter(n.match_key for n in compound_norm)
    multi = sum(1 for v in keys.values() if v > 1)
    rep(f"compound match keys shared by >1 matrix column (e.g. multiple doses/entries; descriptive): {multi}")

    # duplicate identifiers: reported, never aggregated
    dups: list[dict] = []
    for cid, n in Counter(cell_ids).items():
        if n > 1:
            dups.append({"source": "prism_cell_line_id", "id": cid, "n_copies": n, "originals": cid})
    by_key: dict[str, list[str]] = {}
    for orig, nid in zip(compound_original, compound_norm):
        by_key.setdefault(nid.column_key, []).append(orig)
    for k, origs in by_key.items():
        if len(origs) > 1:
            dups.append({"source": "prism_compound_key", "id": k, "n_copies": len(origs),
                         "originals": "|".join(origs)})
    changed = sum(1 for a, b in zip(row_labels_raw if orientation == "cells_by_compounds" else col_labels_raw,
                                    cell_ids) if a != b)
    rep(f"cell-line labels altered by whitespace stripping: {changed}")
    non_ach = [c for c in cell_ids if not re.fullmatch(ACH_PATTERN, c)]
    rep(f"cell-axis labels that are not ACH IDs: {len(non_ach)} {non_ach[:5]}")
    rep(f"duplicated identifiers in matrix: cell lines={sum(1 for d in dups if d['source'] == 'prism_cell_line_id')}, "
        f"compound keys={sum(1 for d in dups if d['source'] == 'prism_compound_key')}")

    rep.summary["prism"] = {
        "orientation": orientation, "cells": len(cell_ids), "compounds": len(compound_original),
        "entry_categories": cc, "normalization_rule_counts": nrm,
        "match_keys_with_multiple_columns": multi, "non_ach_cell_labels": len(non_ach),
        "duplicate_cell_ids": sum(1 for d in dups if d["source"] == "prism_cell_line_id"),
        "duplicate_compound_keys": sum(1 for d in dups if d["source"] == "prism_compound_key"),
    }
    rep.tables["norm_matrix"] = normalization_table(compound_norm, "prism_matrix")
    return PrismMatrix(values, codes, cell_ids, list(compound_original), compound_norm, orientation, dups)


# --------------------------------------------------------------------------- compound annotation
def audit_compound_annotation(p: AuditParams, prism: PrismMatrix, rep: Report):
    rep.section("4. PRISM compound annotation table")
    df = read_csv_checked(p.prism_compounds, "PRISM compound annotation")
    rep(f"shape: {df.shape[0]} x {df.shape[1]} | columns: {[str(c) for c in df.columns]}")
    if df.empty:
        raise AuditError("PRISM compound annotation: table has no rows.")
    matrix_keys = {n.match_key for n in prism.compound_norm}
    if p.compound_id_col:
        require_columns(df, [p.compound_id_col], "PRISM compound annotation", "(requested with --compound-id-col)")
        col = p.compound_id_col
    else:
        scores = {}
        for c in df.columns:
            vals = df[c].dropna().astype(str)
            scores[c] = len({normalize_compound_id(v).match_key for v in vals} & matrix_keys) if len(vals) else 0
        top = max(scores.values()) if scores else 0
        if top == 0:
            raise AuditError(f"PRISM compound annotation: no column matches any matrix compound after "
                             f"normalisation. Columns: {[str(c) for c in df.columns]}. Use --compound-id-col.")
        tied = [c for c, s in scores.items() if s == top]
        if len(tied) > 1:
            raise AuditError(f"PRISM compound annotation: ID column is ambiguous; columns {tied} each match "
                             f"{top} matrix compounds. Choose one with --compound-id-col.")
        col = tied[0]
    rep(f"ID column: {col!r}")
    ann_raw = df[col]
    n_blank = int(ann_raw.isna().sum())
    ann = [normalize_compound_id(v) for v in ann_raw.dropna().astype(str)]
    rep(f"annotation entries: {len(ann)} (blank IDs: {n_blank})")
    nrm = report_rule_counts(rep, ann, "annotation")

    ann_orig = {n.original for n in ann}
    key_counts = Counter(n.match_key for n in ann)
    ann_keys = set(key_counts)
    ann_bases = {n.base_key for n in ann if n.base_key}
    dup_keys = sorted(k for k, v in key_counts.items() if v > 1)
    rep(f"annotation match keys duplicated within the table (ambiguous join, listed in "
        f"compound_match_review.csv): {len(dup_keys)} {dup_keys[:5]}")

    tiers: dict[str, int] = Counter()
    review = []
    for n in prism.compound_norm:
        if n.original in ann_orig:
            tier = "exact_original"
        elif n.match_key in ann_keys:
            tier = "normalized"
        elif n.base_key and n.base_key in ann_bases:
            tier = "base_only_REVIEW"
        else:
            tier = "unmatched"
        tiers[tier] += 1
        ambiguous = n.match_key in dup_keys
        if tier in ("base_only_REVIEW", "unmatched") or ambiguous:
            cands = sorted({a.original for a in ann if n.base_key and a.base_key == n.base_key})[:5]
            review.append({"side": "matrix", "original": n.original, "match_key": n.match_key,
                           "base_key": n.base_key, "status": tier + ("+annotation_duplicated" if ambiguous else ""),
                           "candidates": "|".join(cands)})
    matrix_bases = {n.base_key for n in prism.compound_norm if n.base_key}
    ann_only = [a for a in ann if a.match_key not in matrix_keys]
    ann_only_base = 0
    seen = set()
    for a in ann_only:
        if a.match_key in seen:
            continue
        seen.add(a.match_key)
        is_base = bool(a.base_key and a.base_key in matrix_bases)
        ann_only_base += is_base
        review.append({"side": "annotation", "original": a.original, "match_key": a.match_key,
                       "base_key": a.base_key, "status": "annotation_only" + ("+base_in_matrix_REVIEW" if is_base else ""),
                       "candidates": ""})
    rep("matrix compound columns by match tier (descriptive; only exact_original and normalized are accepted matches):")
    for t in ("exact_original", "normalized", "base_only_REVIEW", "unmatched"):
        rep(f"  {t}: {tiers.get(t, 0)}")
    rep(f"annotation match keys not in matrix: {len(seen)} (of which share a BRD base with a matrix compound: {ann_only_base})")
    rep.summary["compound_annotation"] = {
        "id_col": col, "entries": len(ann), "blank_ids": n_blank, "normalization_rule_counts": nrm,
        "matrix_columns_by_tier": dict(tiers), "annotation_duplicate_match_keys": len(dup_keys),
        "annotation_keys_not_in_matrix": len(seen), "annotation_only_with_base_in_matrix": ann_only_base,
    }
    rep.tables["norm_annotation"] = normalization_table(ann, "annotation")
    rep.tables["match_review"] = pd.DataFrame(
        review, columns=["side", "original", "match_key", "base_key", "status", "candidates"])


def audit_prism_cell_meta(p: AuditParams, prism: PrismMatrix, rep: Report):
    rep.section("5. PRISM cell-line annotation table")
    df = read_csv_checked(p.prism_cell_meta, "PRISM cell-line annotation")
    rep(f"shape: {df.shape[0]} x {df.shape[1]} | columns: {[str(c) for c in df.columns]}")
    col = find_ach_column(df, list(df.columns), None, "PRISM cell-line annotation", "(edit the table or file an issue)")
    ids = df[col].astype(str).str.strip()
    matrix_ids = {c for c in prism.cell_ids if re.fullmatch(ACH_PATTERN, c)}
    rep(f"ID column: {col!r} | entries: {len(ids)} | duplicated IDs: {int(ids.duplicated().sum())}")
    rep(f"in table, not in matrix: {len(set(ids) - matrix_ids)}; in matrix, not in table: {len(matrix_ids - set(ids))}")
    for c in df.columns:
        if "str" in str(c).lower() or "pass" in str(c).lower():
            rep(f"{c!r} value counts: {df[c].value_counts(dropna=False).head(8).to_dict()}")


# --------------------------------------------------------------------------- expression
@dataclass
class ExprResult:
    ids: set
    duplicates: list
    selected_positions: list
    retained_positions: list
    id_col: str
    default_col: str | None
    models_without_default: list


def audit_expression(p: AuditParams, rep: Report) -> ExprResult:
    rep.section("1. Expression matrix")
    df = read_csv_checked(p.expression, "Expression", low_memory=False)
    gene_cols = [c for c in df.columns if GENE_COL_RE.match(str(c))]
    gene_set = set(gene_cols)
    meta_cols = [c for c in df.columns if c not in gene_set]
    rep(f"shape (rows x columns): {df.shape[0]} x {df.shape[1]}")
    rep(f"metadata/ID columns ({len(meta_cols)}): {[str(c) for c in meta_cols]}")
    if not gene_cols:
        raise AuditError(f"Expression: no gene columns matching 'SYMBOL (ID)' were found. First columns: "
                         f"{[str(c) for c in df.columns][:10]}. This file may be a different layout/release.")
    rep(f"gene columns matching 'SYMBOL (ID)': {len(gene_cols)}")

    id_col = find_ach_column(df, meta_cols, p.expression_id_col, "Expression", "--expression-id-col")
    rep(f"model-ID column: {id_col!r}")
    raw_ids = df[id_col].astype(str)
    ids = raw_ids.str.strip()
    rep(f"IDs altered by whitespace stripping: {int((raw_ids != ids).sum())}; "
        f"IDs not matching {ACH_PATTERN}: {int((~ids.str.fullmatch(ACH_PATTERN)).sum())}")
    rep(f"rows: {len(ids)} | unique IDs: {ids.nunique()}")

    # ---- default-entry selection (selection mask first; duplicates evaluated on the SELECTED rows only)
    if p.expression_default_col:
        require_columns(df, [p.expression_default_col], "Expression", "(requested with --expression-default-col)")
        flag_col = p.expression_default_col
    else:
        cands = [c for c in meta_cols if "default" in str(c).lower()]
        if len(cands) > 1:
            raise AuditError(f"Expression: several columns look like default-entry flags {cands}; "
                             f"choose one with --expression-default-col.")
        flag_col = cands[0] if cands else None
    models_without_default: list[str] = []
    if flag_col is not None:
        tok = df[flag_col].astype(str).str.strip().str.lower()
        bad = sorted(set(tok) - TRUE_TOKENS - FALSE_TOKENS)
        if bad:
            raise AuditError(f"Expression: default-entry column {flag_col!r} has unrecognised values {bad}; "
                             f"expected one of {sorted(TRUE_TOKENS | FALSE_TOKENS)}.")
        selected = tok.isin(TRUE_TOKENS).to_numpy()
        rep(f"default-entry flag column {flag_col!r}: value counts {df[flag_col].value_counts(dropna=False).to_dict()}")
        rep(f"rows selected as default: {int(selected.sum())}; rows not selected: {int((~selected).sum())} "
            f"(not selected means not analysed; never used as fallback)")
        models_without_default = sorted(set(ids) - set(ids[selected]))
        rep(f"models with NO default-flagged row (excluded from the expression set, listed in exclusions.csv): "
            f"{len(models_without_default)} {models_without_default[:5]}")
        for m in models_without_default:
            rep.exclusion_rows.append({"source": "expression", "id": m, "reason": "no_default_flagged_row"})
    else:
        selected = np.ones(len(df), dtype=bool)
        rep("no default-entry flag column found: ALL rows are treated as selected; repeated IDs are duplicates.")
    sel_pos = [int(i) for i in np.flatnonzero(selected)]
    sel_ids = ids[selected]
    counts = sel_ids.value_counts()
    dup_ids = sorted(counts[counts > 1].index)
    dups = [{"source": "expression_model_id", "id": i, "n_copies": int(counts[i]), "originals": i} for i in dup_ids]
    rep(f"selected rows: {len(sel_pos)} | unique selected IDs: {sel_ids.nunique()} | "
        f"IDs with >1 SELECTED row (ambiguous; never resolved by 'keep first'): {len(dup_ids)} {dup_ids[:5]}")
    retained = [i for i in sel_pos if ids.iloc[i] not in set(dup_ids)]

    # ---- numeric validation of gene values
    bad_cols = [c for c in gene_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if bad_cols:
        vals, codes, examples = classify_strings(df[bad_cols])
        cc = category_counts(codes)
        rep(f"{len(bad_cols)} gene columns are not numeric dtype; entry categories within them: {cc}")
        if cc["nonnumeric"]:
            ex = [(str(ids.iloc[i]), str(bad_cols[j]), tok) for i, j, tok in examples[:10]]
            if not p.allow_nonnumeric:
                raise AuditError(f"Expression: {cc['nonnumeric']} nonnumeric entries in gene columns. Examples "
                                 f"(model, gene, token): {ex}. Re-run with --allow-nonnumeric to count them as non-usable.")
            rep(f"  nonnumeric entries counted as non-usable (--allow-nonnumeric): examples {ex}")
        df[bad_cols] = vals
    X = df.loc[selected, gene_cols].to_numpy(dtype=np.float32)
    n_nan, n_inf = int(np.isnan(X).sum()), int(np.isinf(X).sum())
    finite = X[np.isfinite(X)]
    rep(f"analysed expression block (selected rows, statistics BEFORE duplicate exclusion): {X.shape[0]} x {X.shape[1]}")
    rep(f"missing/NaN entries: {n_nan} ({n_nan / X.size:.4%}) | infinite entries: {n_inf}")
    if finite.size:
        rep(f"finite value range: min={finite.min():.3f} max={finite.max():.3f}")
    rep(f"negative entries: {int((finite < 0).sum())} (log2(TPM+1) is non-negative; negatives suggest batch-corrected data)")
    sd = np.nanstd(X, axis=0)
    rep(f"zero-variance genes: {int((sd == 0).sum())}")
    symbols = pd.Series([str(c).rsplit(" (", 1)[0] for c in gene_cols])
    rep(f"duplicate gene symbols among columns: {int(symbols.duplicated().sum())}")

    rep.summary["expression"] = {
        "rows": int(df.shape[0]), "columns": int(df.shape[1]), "id_col": str(id_col),
        "default_col": None if flag_col is None else str(flag_col), "n_gene_cols": len(gene_cols),
        "selected_rows": len(sel_pos), "unique_selected_ids": int(sel_ids.nunique()),
        "ids_with_multiple_selected_rows": len(dup_ids), "models_without_default": len(models_without_default),
        "missing_entries": n_nan, "infinite_entries": n_inf,
        "selected_row_positions": sel_pos, "retained_row_positions_before_policy": retained,
    }
    return ExprResult(set(sel_ids), dups, sel_pos, retained, str(id_col),
                      None if flag_col is None else str(flag_col), models_without_default)


# --------------------------------------------------------------------------- model metadata
@dataclass
class ModelResult:
    df: pd.DataFrame
    id_col: str
    duplicates: list
    alias_map: dict       # alias ACH id -> set of current ModelIDs claiming it


def load_model(p: AuditParams, rep: Report) -> ModelResult:
    rep.section("2. Model metadata")
    m = read_csv_checked(p.model, "Model.csv", low_memory=False)
    rep(f"shape: {m.shape[0]} x {m.shape[1]} | columns: {[str(c) for c in m.columns]}")
    require_columns(m, [p.model_id_col], "Model.csv",
                    "If this release renames the column, pass --model-id-col.")
    m = m.copy()
    m[p.model_id_col] = m[p.model_id_col].astype(str).str.strip()
    counts = m[p.model_id_col].value_counts()
    dups = [{"source": "model_csv_model_id", "id": i, "n_copies": int(n), "originals": i}
            for i, n in counts[counts > 1].items()]
    rep(f"unique ModelIDs: {m[p.model_id_col].nunique()} | duplicated ModelIDs: {len(dups)}")
    alias_map: dict[str, set] = {}
    n_unparsed = 0
    if "ModelIDAlias" in m.columns:
        for mid, al in zip(m[p.model_id_col], m["ModelIDAlias"]):
            if isinstance(al, str) and al.strip():
                found = [a for a in ACH_FIND.findall(al) if a != mid]
                if not found:
                    n_unparsed += 1
                for a in found:
                    alias_map.setdefault(a, set()).add(mid)
        rep(f"ModelIDAlias present: {len(alias_map)} distinct alias IDs parsed; non-empty alias cells "
            f"without a parseable ACH ID: {n_unparsed}")
    else:
        rep("ModelIDAlias column not present in this release: alias matching is not possible.")
    rep.summary["model"] = {"rows": int(m.shape[0]), "duplicated_model_ids": len(dups),
                            "alias_ids": len(alias_map), "alias_cells_unparsed": n_unparsed}
    return ModelResult(m, p.model_id_col, dups, alias_map)


# --------------------------------------------------------------------------- overlap and usable observations
def audit_overlap(p: AuditParams, expr_ids: set, prism: PrismMatrix, model: ModelResult, rep: Report):
    rep.section("6. Identifier overlap: DIRECT matches vs alias candidates")
    prism_ach = [c for c in prism.cell_ids if re.fullmatch(ACH_PATTERN, c)]
    prism_set = set(prism_ach)
    direct = sorted(expr_ids & prism_set)
    prism_only, expr_only = prism_set - expr_ids, expr_ids - prism_set
    non_ach = [c for c in prism.cell_ids if not re.fullmatch(ACH_PATTERN, c)]
    rep(f"expression IDs: {len(expr_ids)} | PRISM cell-line IDs (ACH): {len(prism_set)} | PRISM non-ACH labels (never matched): {len(non_ach)}")
    rep(f"DIRECT identifier overlap (the only set used for all counts below): {len(direct)}")
    rep(f"PRISM-only: {len(prism_only)} | expression-only: {len(expr_only)}")
    model_ids = set(model.df[model.id_col])
    rep(f"PRISM IDs absent from Model.csv: {len(prism_set - model_ids)} | expression IDs absent from Model.csv: {len(expr_ids - model_ids)}")
    if non_ach:
        for c in non_ach:
            rep.exclusion_rows.append({"source": "prism_matrix", "id": c, "reason": "non_ACH_cell_axis_label_not_matchable"})

    alias_rows = []
    def add_alias(direction, side_id, targets, other_set, own_set):
        tl = sorted(targets)
        if len(tl) > 1:
            status = "ambiguous_multiple_targets"
        else:
            t = tl[0]
            if t not in other_set:
                status = "target_not_in_other_dataset"
            elif t in own_set:
                status = "target_also_present_directly_MERGE_WOULD_DUPLICATE"
            else:
                status = "reviewable"
        alias_rows.append({"direction": direction, "id": side_id, "candidate_model_ids": "|".join(tl), "status": status})
    for pid in sorted(prism_only):
        if pid in model.alias_map:
            add_alias("prism_id_is_alias_of_expression_model", pid, model.alias_map[pid], expr_ids, prism_set)
    for eid in sorted(expr_only):
        if eid in model.alias_map:
            add_alias("expression_id_is_alias_of_prism_model", eid, model.alias_map[eid], prism_set, expr_ids)
    alias_df = pd.DataFrame(alias_rows, columns=["direction", "id", "candidate_model_ids", "status"])
    rep.tables["alias_candidates"] = alias_df
    st = alias_df["status"].value_counts().to_dict() if len(alias_df) else {}
    rep(f"alias candidates (NOT merged; NOT included in any count; see alias_candidates.csv): {len(alias_df)} {st}")
    reviewable = int((alias_df["status"] == "reviewable").sum()) if len(alias_df) else 0
    rep(f"  hypothetical extra lines IF every 'reviewable' candidate were manually accepted: {reviewable}")
    rep.summary["overlap"] = {
        "expression_ids": len(expr_ids), "prism_ach_ids": len(prism_set), "prism_non_ach_labels": len(non_ach),
        "direct_overlap": len(direct), "prism_only": len(prism_only), "expression_only": len(expr_only),
        "alias_candidates": len(alias_df), "alias_candidate_status_counts": st,
        "alias_reviewable_not_merged": reviewable,
    }
    if not direct:
        rep("No direct overlap: the two files are not ID-compatible as supplied. Stopping before usable-observation counts.")
        return

    rep.section("7. Usable observations (DIRECT-overlap lines only)")
    direct_set = set(direct)
    rows = [i for i, c in enumerate(prism.cell_ids) if c in direct_set]
    sv, sc = prism.values[rows], prism.codes[rows]
    cc = category_counts(sc)
    rep(f"overlap submatrix: {sv.shape[0]} lines x {sv.shape[1]} compound columns = {cc['total']:,} entries")
    for k in ("finite_numeric", "missing", "nonfinite", "nonnumeric"):
        rep(f"  {k}: {cc[k]:,} ({cc[k] / cc['total']:.2%})" if cc["total"] else f"  {k}: 0")
    rep(f"usable observations = finite_numeric entries = {cc['finite_numeric']:,}; "
        f"missing + nonfinite + nonnumeric = {cc['missing'] + cc['nonfinite'] + cc['nonnumeric']:,}; sum check = {cc['total']:,}")
    n_lines = (sc == FINITE).sum(axis=0)
    n_cmp = (sc == FINITE).sum(axis=1)
    rep(f"finite observations per compound column, quantiles: {json.dumps(qdict(n_lines))}")
    rep(f"finite observations per cell line, quantiles: {json.dumps(qdict(n_cmp))}")
    rep(f"cell lines with zero finite observations: {int((n_cmp == 0).sum())}")
    n_ov = len(rows)
    thr = sorted({int(math.ceil(f * n_ov)) for f in p.coverage_fractions} | {int(t) for t in p.coverage_thresholds})
    cov = {f">={t} finite lines": int((n_lines >= t).sum()) for t in thr}
    rep(f"compound columns by coverage (thresholds derived from the supplied fractions of {n_ov} overlap lines "
        f"and any absolute thresholds; descriptive only): {cov}")
    sdf = pd.DataFrame(sv)
    sd = sdf.std(axis=0)
    rep(f"per-compound SD quantiles: {json.dumps({k: round(v, 4) for k, v in qdict(sd).items()})}")
    for t in p.sd_thresholds:
        rep(f"  compound columns with SD < {t}: {int((sd < t).sum())}")
    rep.tables["coverage"] = pd.DataFrame({
        "compound_original": prism.compound_original,
        "column_key": [n.column_key for n in prism.compound_norm],
        "n_finite": n_lines, "n_missing": (sc == MISSING).sum(axis=0),
        "n_nonfinite": (sc == NONFINITE).sum(axis=0), "n_nonnumeric": (sc == NONNUMERIC).sum(axis=0),
        "mean_response": sdf.mean(axis=0).to_numpy(), "sd_response": sd.to_numpy()})
    rep.summary["overlap"].update({
        "usable_observations": cc["finite_numeric"], "entry_categories_overlap": cc,
        "finite_per_compound_quantiles": qdict(n_lines), "compound_columns_by_coverage": cov})

    rep.section("8. Confounder diagnostics (descriptive)")
    line_median = pd.Series(np.nanmedian(np.where(np.isfinite(sv), sv, np.nan), axis=1)
                            if sv.size else [], index=[prism.cell_ids[i] for i in rows])
    rep(f"per-line median response across compounds, quantiles: {json.dumps({k: round(v, 4) for k, v in qdict(line_median).items()})}")
    mm = model.df
    mm = mm[~mm[model.id_col].isin({d['id'] for d in model.duplicates})].set_index(model.id_col)
    lineage_col = p.lineage_col
    if lineage_col:
        require_columns(mm.reset_index(), [lineage_col], "Model.csv", "(requested with --lineage-col)")
    else:
        lineage_col = next((c for c in ("OncotreeLineage", "OncotreePrimaryDisease") if c in mm.columns), None)
    if lineage_col:
        lin = mm.reindex(direct)[lineage_col]
        tab = lin.value_counts(dropna=False).rename_axis(str(lineage_col)).reset_index(name="n_overlap_lines")
        rep.tables["lineage"] = tab
        rep(f"lineage column: {lineage_col!r}; groups: {lin.nunique(dropna=True)}; lines with missing lineage: {int(lin.isna().sum())}")
        rep(f"smallest 5 groups: {tab.tail(5).to_dict('records')}")
        eta = eta_squared(line_median, pd.Series(lin.reindex(line_median.index).to_numpy(), index=line_median.index))
        rep(f"share of variance in per-line median response explained by {lineage_col}: {None if eta is None else round(eta, 4)}")
        rep.summary["overlap"]["lineage_eta_squared"] = eta
    else:
        rep("No lineage column found in Model.csv (looked for OncotreeLineage, OncotreePrimaryDisease); use --lineage-col.")
    if "PatientID" in mm.columns:
        pc = mm.reindex(direct)["PatientID"].dropna().value_counts()
        rep(f"PatientID: {int((pc > 1).sum())} patients contribute >1 direct-overlap line (use grouped splits)")
        rep.summary["overlap"]["patients_with_multiple_lines"] = int((pc > 1).sum())
    if p.omics_profiles:
        op = read_csv_checked(p.omics_profiles, "OmicsProfiles", low_memory=False)
        require_columns(op, ["ModelID"], "OmicsProfiles")
        op = op[op["ModelID"].isin(direct)]
        for c in op.columns:
            if any(k in str(c).lower() for k in ("strand", "source", "library")):
                rep(f"{c!r} among direct-overlap profiles: {op[c].value_counts(dropna=False).head(8).to_dict()}")


# --------------------------------------------------------------------------- orchestration
def audit_provenance(p: AuditParams, rep: Report):
    rep.section("0. Provenance")
    prov = {}
    for label in ("expression", "model", "prism_matrix", "prism_compounds", "prism_cell_meta", "omics_profiles"):
        path = getattr(p, label)
        if path is None:
            continue
        path = Path(path)
        if not path.is_file():
            raise AuditError(f"{label}: file not found: {path}")
        prov[label] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        rep(f"{label:16s} {path.name}  {prov[label]['bytes']:,} bytes  sha256={prov[label]['sha256']}")
    rep(f"python={platform.python_version()} pandas={pd.__version__} numpy={np.__version__}")
    rep.summary["provenance"] = prov
    rep.summary["versions"] = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}


def _run(p: AuditParams, rep: Report, outdir: Path):
    if p.duplicate_policy not in ("fail", "exclude"):
        raise AuditError("--duplicate-policy must be 'fail' or 'exclude'.")
    rep("DESCRIPTIVE AUDIT: counts describe the supplied files only. Exclusions are listed in exclusions.csv.")
    audit_provenance(p, rep)
    expr = audit_expression(p, rep)
    model = load_model(p, rep)
    prism = load_prism_matrix(p, rep)
    if p.prism_compounds:
        audit_compound_annotation(p, prism, rep)
    if p.prism_cell_meta:
        audit_prism_cell_meta(p, prism, rep)

    rep.section("Integrity gate: duplicated identifiers")
    dups = expr.duplicates + model.duplicates + prism.duplicates
    rep.summary["duplicates"] = {"total_ambiguous_ids": len(dups), "policy": p.duplicate_policy}
    if dups:
        ddf = pd.DataFrame(dups)
        rep.tables["duplicates_report"] = ddf
        rep(f"{len(dups)} duplicated identifiers across files (see duplicates_report.csv):")
        for d in dups[:20]:
            rep(f"  {d['source']}: {d['id']} x{d['n_copies']} originals={d['originals']}")
        if p.duplicate_policy == "fail":
            rep("Policy 'fail': overlap and usable-observation counts were NOT computed. Fix the duplicates, or re-run with "
                "--duplicate-policy exclude to exclude every copy of each ambiguous ID (nothing is kept or aggregated).")
            raise IntegrityBlock(f"{len(dups)} duplicated identifiers; see duplicates_report.csv")
        rep("Policy 'exclude': EVERY copy of each ambiguous ID is excluded below. No row is kept, averaged or aggregated.")
        for d in dups:
            rep.exclusion_rows.append({"source": d["source"], "id": d["id"],
                                       "reason": f"duplicate_identifier_x{d['n_copies']}_excluded_by_policy"})
        e_ids = {d["id"] for d in expr.duplicates}
        expr.ids -= e_ids
        m_ids = {d["id"] for d in model.duplicates}
        model.df = model.df[~model.df[model.id_col].isin(m_ids)]
        prism.drop({d["id"] for d in prism.duplicates if d["source"] == "prism_cell_line_id"},
                   {d["id"] for d in prism.duplicates if d["source"] == "prism_compound_key"})
        rep.summary["exclusions"]["duplicate_ids_excluded"] = {
            "expression": len(e_ids), "model_csv": len(m_ids),
            "prism_cells": sum(1 for d in prism.duplicates if d["source"] == "prism_cell_line_id"),
            "prism_compounds": sum(1 for d in prism.duplicates if d["source"] == "prism_compound_key")}
    else:
        rep("No duplicated identifiers in the selected expression rows, Model.csv, or the PRISM matrix.")

    audit_overlap(p, expr.ids, prism, model, rep)
    rep.summary["exclusions"]["models_without_default_row"] = len(expr.models_without_default)
    rep.summary["exclusions"]["prism_non_ach_labels"] = rep.summary.get("overlap", {}).get("prism_non_ach_labels")


def write_outputs(rep: Report, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "audit_report.txt").write_text("\n".join(rep.lines), encoding="utf-8")
    (outdir / "audit_summary.json").write_text(json.dumps(rep.summary, indent=2, default=jsonable), encoding="utf-8")
    names = {"duplicates_report": "duplicates_report.csv", "alias_candidates": "alias_candidates.csv",
             "coverage": "compound_coverage.csv", "lineage": "overlap_by_lineage.csv",
             "match_review": "compound_match_review.csv"}
    for key, fname in names.items():
        if key in rep.tables:
            rep.tables[key].to_csv(outdir / fname, index=False)
    norms = [rep.tables[k] for k in ("norm_matrix", "norm_annotation") if k in rep.tables]
    if norms:
        pd.concat(norms).to_csv(outdir / "compound_id_normalization.csv", index=False)
    pd.DataFrame(rep.exclusion_rows, columns=["source", "id", "reason"]).to_csv(outdir / "exclusions.csv", index=False)


def run_audit(p: AuditParams) -> dict:
    """Run the audit. Raises AuditError / IntegrityBlock; outputs are written in every case."""
    rep = Report(p.quiet)
    outdir = Path(p.outdir)
    try:
        _run(p, rep, outdir)
        rep.summary["status"] = "complete"
    except IntegrityBlock:
        rep.summary["status"] = "blocked_by_duplicated_identifiers"
        raise
    except AuditError as e:
        rep.summary["status"] = "error"
        rep.summary["error"] = str(e)
        rep(f"\nERROR: {e}")
        raise
    finally:
        write_outputs(rep, outdir)
    return rep.summary


def parse_args(argv=None) -> AuditParams:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expression", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--prism-matrix", type=Path, required=True)
    ap.add_argument("--prism-compounds", type=Path)
    ap.add_argument("--prism-cell-meta", type=Path)
    ap.add_argument("--omics-profiles", type=Path)
    ap.add_argument("--prism-orientation", default="auto", choices=["auto", "cells_by_compounds", "compounds_by_cells"])
    ap.add_argument("--expression-id-col")
    ap.add_argument("--expression-default-col")
    ap.add_argument("--model-id-col", default="ModelID")
    ap.add_argument("--compound-id-col")
    ap.add_argument("--lineage-col")
    ap.add_argument("--duplicate-policy", default="fail", choices=["fail", "exclude"])
    ap.add_argument("--allow-nonnumeric", action="store_true")
    ap.add_argument("--coverage-fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.9, 1.0],
                    help="fractions of overlap lines used to derive descriptive coverage thresholds")
    ap.add_argument("--coverage-thresholds", type=int, nargs="+", default=[],
                    help="additional absolute line-count thresholds (descriptive)")
    ap.add_argument("--sd-thresholds", type=float, nargs="+", default=[],
                    help="optional SD values for descriptive counts (none are applied by default)")
    ap.add_argument("--outdir", type=Path, default=Path("results/audit"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    return AuditParams(
        expression=a.expression, model=a.model, prism_matrix=a.prism_matrix, outdir=a.outdir,
        prism_compounds=a.prism_compounds, prism_cell_meta=a.prism_cell_meta, omics_profiles=a.omics_profiles,
        prism_orientation=a.prism_orientation, expression_id_col=a.expression_id_col,
        expression_default_col=a.expression_default_col, model_id_col=a.model_id_col,
        compound_id_col=a.compound_id_col, lineage_col=a.lineage_col, duplicate_policy=a.duplicate_policy,
        allow_nonnumeric=a.allow_nonnumeric, coverage_fractions=tuple(a.coverage_fractions),
        coverage_thresholds=tuple(a.coverage_thresholds), sd_thresholds=tuple(a.sd_thresholds), quiet=a.quiet)


def main(argv=None) -> int:
    params = parse_args(argv)
    try:
        run_audit(params)
        return 0
    except IntegrityBlock as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 3
    except AuditError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
