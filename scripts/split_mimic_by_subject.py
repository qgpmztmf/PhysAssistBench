"""
split_mimic_by_subject.py — Partition raw MIMIC-IV CSVs into per-subject files.

PhysAssistBench reads patient data as one directory per subject_id:

    <out_root>/<subject_id>/<module>_<table>.csv.csv.gz

e.g. raw_data/MIMIC-IV-split-by-patient/split_data_each_patient/10000032/hosp_labevents.csv.csv.gz

This script consumes the raw MIMIC-IV download (standard PhysioNet layout) and
produces that structure. Each source module directory is given a prefix; every
`*.csv.gz` table in it that has a `subject_id` column is split by subject_id and
written to `<module>_<table>.csv.csv.gz` inside each subject's directory. Tables
without a `subject_id` column (dimension tables such as d_labitems,
d_icd_diagnoses, d_items) are skipped — they belong in MIMIC_REF_ROOT, not the
per-subject split (copy them there separately; see README).

Large tables are streamed in chunks; per-subject files are written as
concatenated gzip members (transparently read back by pandas), so memory stays
bounded by --chunksize and only the final compressed output is kept on disk.

Standard MIMIC-IV layout (paths are examples; pass your own):
    mimiciv/3.1/hosp/*.csv.gz            -> prefix hosp
    mimiciv/3.1/icu/*.csv.gz             -> prefix icu
    mimiciv/ed-2.2/ed/*.csv.gz           -> prefix ed
    mimic-iv-note/2.2/note/*.csv.gz      -> prefix note

Usage:
    python scripts/split_mimic_by_subject.py \
        --out_root raw_data/MIMIC-IV-split-by-patient/split_data_each_patient \
        --source hosp /path/to/mimiciv/3.1/hosp \
        --source icu  /path/to/mimiciv/3.1/icu \
        --source ed   /path/to/mimiciv/ed-2.2/ed \
        --source note /path/to/mimic-iv-note/2.2/note

    # Only split a subset of subjects (e.g. to build a small sample):
    python scripts/split_mimic_by_subject.py ... --subjects 10000032 10000048
    python scripts/split_mimic_by_subject.py ... --subjects_file subjects.txt

Note: splitting the full MIMIC-IV is I/O heavy and can take hours; labevents and
chartevents dominate. Use --subjects to build a sample quickly.
"""

import argparse
import glob
import gzip
import os
import sys

import pandas as pd


def _read_header(path: str) -> list[str]:
    with gzip.open(path, "rt") as f:
        first = f.readline().strip()
    return first.split(",") if first else []


def split_table(
    src_path: str,
    prefix: str,
    out_root: str,
    chunksize: int,
    subjects: set[int] | None,
    written: set[tuple[int, str]],
) -> int:
    """Split one MIMIC table into per-subject gzip members. Returns rows written."""
    table = os.path.basename(src_path)
    for suffix in (".csv.gz", ".csv"):
        if table.endswith(suffix):
            table = table[: -len(suffix)]
            break
    out_name = f"{prefix}_{table}.csv.csv.gz"

    header = _read_header(src_path)
    if "subject_id" not in header:
        print(f"  skip {prefix}/{table}: no subject_id column")
        return 0

    rows = 0
    reader = pd.read_csv(src_path, compression="gzip", chunksize=chunksize,
                         low_memory=False)
    for chunk in reader:
        chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce")
        chunk = chunk.dropna(subset=["subject_id"])
        chunk["subject_id"] = chunk["subject_id"].astype("int64")
        if subjects is not None:
            chunk = chunk[chunk["subject_id"].isin(subjects)]
        if chunk.empty:
            continue
        for sid, group in chunk.groupby("subject_id", sort=False):
            sid = int(sid)
            sdir = os.path.join(out_root, str(sid))
            os.makedirs(sdir, exist_ok=True)
            key = (sid, out_name)
            first = key not in written
            with gzip.open(os.path.join(sdir, out_name), "at", newline="") as f:
                group.to_csv(f, header=first, index=False)
            written.add(key)
            rows += len(group)
    print(f"  {prefix}/{table}: wrote {rows} rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_root", required=True,
                        help="Destination split_data_each_patient/ directory")
    parser.add_argument("--source", nargs=2, action="append", metavar=("PREFIX", "DIR"),
                        required=True, help="Module prefix and its directory of *.csv.gz tables (repeatable)")
    parser.add_argument("--chunksize", type=int, default=1_000_000,
                        help="Rows per read chunk (default 1e6)")
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="Only split these subject_ids")
    parser.add_argument("--subjects_file", type=str, default=None,
                        help="File with one subject_id per line (union with --subjects)")
    args = parser.parse_args()

    subjects: set[int] | None = None
    if args.subjects or args.subjects_file:
        subjects = set(args.subjects or [])
        if args.subjects_file:
            with open(args.subjects_file) as f:
                subjects |= {int(x) for x in f.read().split() if x.strip()}
        print(f"Restricting to {len(subjects)} subject(s)")

    os.makedirs(args.out_root, exist_ok=True)
    written: set[tuple[int, str]] = set()
    total = 0
    for prefix, d in args.source:
        tables = sorted(glob.glob(os.path.join(d, "*.csv.gz")))
        if not tables:
            print(f"[{prefix}] no *.csv.gz in {d}", file=sys.stderr)
            continue
        print(f"[{prefix}] {len(tables)} table(s) in {d}")
        for t in tables:
            total += split_table(t, prefix, args.out_root, args.chunksize, subjects, written)

    n_subjects = len({k[0] for k in written})
    print(f"\nDone: {total} rows across {n_subjects} subject director(ies) -> {args.out_root}")


if __name__ == "__main__":
    main()
