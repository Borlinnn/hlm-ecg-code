#!/usr/bin/env python3
"""Download PTB-XL v1.0.3 files required for HLM-ECG.

Default behavior downloads:
- ptbxl_database.csv
- scp_statements.csv
- LICENSE.txt
- RECORDS
- SHA256SUMS.txt
- example_physionet.py
- records100 only: *.hea and *.dat for every filename_lr entry in
  ptbxl_database.csv

It intentionally does not download records500.

Examples
--------
Dry run:
    python scripts/download_ptbxl_records100.py --root data/ptb-xl --dry-run

Pure Python downloader:
    python scripts/download_ptbxl_records100.py --root data/ptb-xl --method python --workers 8 --verify

Wget downloader:
    python scripts/download_ptbxl_records100.py --root data/ptb-xl --method wget --verify

AWS downloader:
    python scripts/download_ptbxl_records100.py --root data/ptb-xl --method aws --verify
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

BASE_URL_TEMPLATE = "https://physionet.org/files/ptb-xl/{version}"
S3_PREFIX_TEMPLATE = "s3://physionet-open/ptb-xl/{version}"
EXPECTED_METADATA_ROWS = 21799
TOP_FILES = [
    "ptbxl_database.csv",
    "scp_statements.csv",
    "LICENSE.txt",
    "RECORDS",
    "SHA256SUMS.txt",
    "example_physionet.py",
]


def log(msg: str) -> None:
    print(f"[download_ptbxl] {msg}", flush=True)


def url_join(base: str, rel: str) -> str:
    return base.rstrip("/") + "/" + rel.lstrip("/")


def download_url(url: str, dest: Path, overwrite: bool = False, retries: int = 3, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HLM-ECG-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response, open(tmp, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            wait = min(2 ** attempt, 20)
            log(f"retry {attempt}/{retries} failed for {url}: {e}; waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {url} -> {dest}: {last_err}")


def download_top_files_python(root: Path, base_url: str, overwrite: bool, dry_run: bool) -> None:
    for name in TOP_FILES:
        url = url_join(base_url, name)
        dest = root / name
        if dry_run:
            print(f"DOWNLOAD {url} -> {dest}")
        else:
            log(f"downloading {name}")
            download_url(url, dest, overwrite=overwrite)


def read_filename_lr_entries(metadata_file: Path) -> List[str]:
    """Read authoritative low-resolution record paths from ptbxl_database.csv."""
    if not metadata_file.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_file}")

    with metadata_file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "filename_lr" not in reader.fieldnames:
            raise RuntimeError(f"Missing filename_lr column in {metadata_file}")

        entries = [str(row["filename_lr"]).strip() for row in reader]

    if len(entries) != EXPECTED_METADATA_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_METADATA_ROWS} filename_lr entries in {metadata_file}, "
            f"found {len(entries)}."
        )

    seen = set()
    duplicates = []
    for entry in entries:
        if not entry.startswith("records100/") or not entry.endswith("_lr"):
            raise RuntimeError(
                f"Unexpected filename_lr outside records100 low-resolution scope: {entry}"
            )
        if entry in seen:
            duplicates.append(entry)
        seen.add(entry)
    if duplicates:
        raise RuntimeError(f"Duplicate filename_lr entries found: {duplicates[:10]}")

    return entries


def read_records100_entries(records_file: Path) -> List[str]:
    """Read records100 entries from RECORDS for auxiliary logging only."""
    if not records_file.exists():
        raise FileNotFoundError(f"Missing RECORDS file: {records_file}")
    entries: List[str] = []
    for raw in records_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # PTB-XL RECORDS normally lists record base paths without extension.
        # Keep only low-resolution records100 entries.
        if line.startswith("records100/") and line.endswith("_lr"):
            entries.append(line)
    if not entries:
        raise RuntimeError(
            "No records100 entries found in RECORDS. "
            "Check that this is the PTB-XL v1.0.3 RECORDS file."
        )
    return entries


def missing_filename_lr_pairs(root: Path, entries: Sequence[str]) -> List[str]:
    missing = []
    for rec_base in entries:
        record = root / rec_base
        if not record.with_suffix(".hea").exists() or not record.with_suffix(".dat").exists():
            missing.append(rec_base)
    return missing


def download_record_pair(args: Tuple[str, str, Path, bool]) -> Tuple[str, bool, str]:
    base_url, rec_base, root, overwrite = args
    try:
        for ext in (".hea", ".dat"):
            rel = rec_base + ext
            url = url_join(base_url, rel)
            dest = root / rel
            download_url(url, dest, overwrite=overwrite, retries=3, timeout=120)
        return rec_base, True, ""
    except Exception as e:  # noqa: BLE001
        return rec_base, False, str(e)


def download_records100_python(root: Path, base_url: str, overwrite: bool, dry_run: bool, workers: int) -> List[str]:
    entries = read_filename_lr_entries(root / "ptbxl_database.csv")
    log(f"authoritative filename_lr entries found in metadata: {len(entries)}")

    records_file = root / "RECORDS"
    if records_file.exists():
        aux_entries = read_records100_entries(records_file)
        log(
            "auxiliary RECORDS records100 entries found: "
            f"{len(aux_entries)}; metadata remains source of truth"
        )

    missing_before = missing_filename_lr_pairs(root, entries)
    if not dry_run:
        log(f"missing/incomplete filename_lr pairs before download: {len(missing_before)}")

    if dry_run:
        for rec in entries[:5]:
            print(f"DOWNLOAD {url_join(base_url, rec + '.hea')} -> {root / (rec + '.hea')}")
            print(f"DOWNLOAD {url_join(base_url, rec + '.dat')} -> {root / (rec + '.dat')}")
        print(f"... total required metadata record pairs: {len(entries)}")
        return []

    tasks = [(base_url, rec, root, overwrite) for rec in entries]
    failures: List[Tuple[str, str]] = []
    done = 0
    with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for rec_base, ok, err in ex.map(download_record_pair, tasks):
            done += 1
            if not ok:
                failures.append((rec_base, err))
            if done % 500 == 0 or done == len(tasks):
                log(f"downloaded/checked {done}/{len(tasks)} record pairs")
    if failures:
        msg = "\n".join([f"{rec}: {err}" for rec, err in failures[:20]])
        raise RuntimeError(f"Failed to download {len(failures)} record pairs. First failures:\n{msg}")
    return missing_before


def run_cmd(cmd: Sequence[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(list(cmd), check=True)


def download_with_wget(root: Path, version: str, dry_run: bool) -> None:
    base = BASE_URL_TEMPLATE.format(version=version)
    root.mkdir(parents=True, exist_ok=True)
    for name in TOP_FILES:
        run_cmd(["wget", "-c", url_join(base, name), "-O", str(root / name)], dry_run=dry_run)
    run_cmd([
        "wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3", "-R", "index.html*",
        url_join(base, "records100/")
    ], dry_run=dry_run)


def download_with_aws(root: Path, version: str, dry_run: bool) -> None:
    prefix = S3_PREFIX_TEMPLATE.format(version=version)
    root.mkdir(parents=True, exist_ok=True)
    for name in TOP_FILES:
        run_cmd(["aws", "s3", "cp", "--no-sign-request", f"{prefix}/{name}", str(root / name)], dry_run=dry_run)
    run_cmd(["aws", "s3", "sync", "--no-sign-request", f"{prefix}/records100", str(root / "records100")], dry_run=dry_run)


def verify_waveform(root: Path, rec_base: str) -> None:
    import wfdb  # type: ignore

    record_path = root / rec_base
    sig, fields = wfdb.rdsamp(str(record_path))
    shape = tuple(sig.shape)
    fs = int(fields.get("fs", -1))
    if shape != (1000, 12):
        raise RuntimeError(f"{rec_base} waveform shape is {shape}, expected (1000, 12)")
    if fs != 100:
        raise RuntimeError(f"{rec_base} sampling rate is {fs}, expected 100")
    log(f"wfdb verified {rec_base}: shape={shape}, fs={fs}")


def verify_download(root: Path, repaired_records: Sequence[str] = ()) -> None:
    """Verify all metadata filename_lr pairs and run the local check script."""
    entries = read_filename_lr_entries(root / "ptbxl_database.csv")
    hea_count = len(list((root / "records100").rglob("*_lr.hea")))
    dat_count = len(list((root / "records100").rglob("*_lr.dat")))
    missing = missing_filename_lr_pairs(root, entries)

    log(f"verification metadata rows: {len(entries)}")
    log(f"verification records100 *_lr.hea count: {hea_count}")
    log(f"verification records100 *_lr.dat count: {dat_count}")
    log(f"verification missing filename_lr pairs: {len(missing)}")

    if hea_count != EXPECTED_METADATA_ROWS or dat_count != EXPECTED_METADATA_ROWS or missing:
        examples = ", ".join(missing[:20])
        raise RuntimeError(
            "PTB-XL records100 verification failed: expected "
            f"{EXPECTED_METADATA_ROWS} .hea and .dat files from metadata filename_lr; "
            f"found {hea_count} .hea, {dat_count} .dat, missing {len(missing)} pairs. "
            f"Examples: {examples}"
        )

    waveform_records = []
    if entries:
        waveform_records.append(entries[0])
    for rec in repaired_records:
        if rec not in waveform_records:
            waveform_records.append(rec)
    for rec in waveform_records:
        verify_waveform(root, rec)

    script = Path("scripts/check_ptbxl_download.py")
    if script.exists():
        cmd = [sys.executable, str(script), "--root", str(root), "--out", "outputs/day1_audit"]
        log("running verification: " + " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        log("check script not found; skipping verification")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PTB-XL v1.0.3 records100 for HLM-ECG.")
    parser.add_argument("--root", type=Path, default=Path("data/ptb-xl"), help="Destination PTB-XL root.")
    parser.add_argument("--version", default="1.0.3", help="PTB-XL version. Default: 1.0.3.")
    parser.add_argument("--method", choices=["python", "wget", "aws"], default="python", help="Download method.")
    parser.add_argument("--workers", type=int, default=8, help="Python downloader parallel workers.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading.")
    parser.add_argument("--metadata-only", action="store_true", help="Download only metadata and auxiliary files.")
    parser.add_argument("--verify", action="store_true", help="Run scripts/check_ptbxl_download.py after download.")
    args = parser.parse_args()

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    base_url = BASE_URL_TEMPLATE.format(version=args.version)

    log(f"destination root: {root}")
    log(f"version: {args.version}")
    log(f"method: {args.method}")
    log("scope: records100 only; records500 is intentionally excluded")

    repaired_records: List[str] = []
    if args.method == "python":
        download_top_files_python(root, base_url, overwrite=args.overwrite, dry_run=args.dry_run)
        if not args.metadata_only:
            if args.dry_run and not (root / "ptbxl_database.csv").exists():
                print("DRY-RUN NOTE: ptbxl_database.csv is not local yet, so filename_lr entries cannot be enumerated.")
                print(f"After metadata exists, records100 .hea/.dat files will be downloaded from {base_url}/records100/")
            else:
                repaired_records = download_records100_python(
                    root,
                    base_url,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    workers=args.workers,
                )
    elif args.method == "wget":
        download_with_wget(root, args.version, dry_run=args.dry_run)
    elif args.method == "aws":
        download_with_aws(root, args.version, dry_run=args.dry_run)
    else:
        raise ValueError(args.method)

    if args.verify and not args.dry_run:
        verify_download(root, repaired_records=repaired_records)

    log("done")


if __name__ == "__main__":
    main()
