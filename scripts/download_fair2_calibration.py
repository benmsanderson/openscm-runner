"""
Download a FaIR 2.x calibration bundle from Zenodo.

Defaults to record 18828694 ("fair calibration data" v1.6.0, published
2026-03-02), the AR7-relevant Smith calibration. The large
``fair_calibrate.zip`` reproducibility archive (~2.9 GB) is skipped
unless ``--include-zip`` is passed; only the CSV files needed by the
FaIRv2 adapter (~2 MB total) are downloaded by default.

Files are written under the target directory with their canonical
(unversioned) names so the adapter's :class:`NativeFairCalibration`
loader finds them. After running:

    scripts/download_fair2_calibration.py

point the demo at the result:

    export FAIR2_CALIBRATION_PATH=$PWD/configurations/fair-calibrate-v1.6.0
    python scripts/demo_fair2.py

To pick a different Zenodo record (e.g. a newer calibration release):

    scripts/download_fair2_calibration.py --record-id 12345678 \\
        --output-dir configurations/fair-calibrate-newer

No external dependencies. Stdlib urllib only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ZENODO_API_BASE = "https://zenodo.org/api/records"

# Default record: v1.6.0 of "fair calibration data" (Smith et al.)
DEFAULT_RECORD_ID = 18828694
DEFAULT_VERSION_LABEL = "1.6.0"

# Files the FaIRv2 adapter understands. The bundle ships extras (we
# download them so the bundle is complete) but anything else is also
# fine; nothing here is a hard exclude. Only the large reproducibility
# zip is skipped by default to save bandwidth.
_SKIP_BY_DEFAULT = {"fair_calibrate.zip"}


def fetch_record_metadata(record_id: int) -> dict:
    url = f"{ZENODO_API_BASE}/{record_id}"
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req) as resp:  # noqa: S310 (hardcoded Zenodo host)
        return json.loads(resp.read())


def download_file(url: str, dest: Path) -> int:
    """Download ``url`` to ``dest``, return bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"Accept": "*/*"})
    written = 0
    with urlopen(req) as resp, dest.open("wb") as fh:  # noqa: S310
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            written += len(chunk)
    return written


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--record-id",
        type=int,
        default=DEFAULT_RECORD_ID,
        help=(
            f"Zenodo record ID (default {DEFAULT_RECORD_ID}, "
            f"v{DEFAULT_VERSION_LABEL})"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write CSVs into. Default is "
            "configurations/fair-calibrate-v<version>/ relative to the "
            "repo root, with <version> taken from the record metadata."
        ),
    )
    p.add_argument(
        "--include-zip",
        action="store_true",
        help=(
            "Also download fair_calibrate.zip (~2.9 GB reproducibility "
            "archive). Off by default; the CSVs are enough for the adapter."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    args = p.parse_args()

    print(f"Fetching Zenodo record {args.record_id} metadata...")
    metadata = fetch_record_metadata(args.record_id)
    version = metadata.get("metadata", {}).get("version", "unknown")
    title = metadata.get("metadata", {}).get("title", "(unknown title)")
    pub_date = metadata.get("metadata", {}).get(
        "publication_date", "(unknown date)"
    )

    output_dir = args.output_dir or Path(__file__).parent.parent / (
        f"configurations/fair-calibrate-v{version}"
    )

    print(f"  title:           {title}")
    print(f"  version:         {version}")
    print(f"  publication:     {pub_date}")
    print(f"  destination:     {output_dir}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    files = metadata.get("files", [])
    if not files:
        print("ERROR: no files in record metadata", file=sys.stderr)
        return 1

    total_downloaded = 0
    skipped_zip = False
    skipped_existing = 0
    for entry in files:
        filename = entry["key"]
        if filename in _SKIP_BY_DEFAULT and not args.include_zip:
            print(
                f"  SKIP {filename} (use --include-zip to download "
                f"{human_bytes(entry.get('size', 0))})"
            )
            skipped_zip = True
            continue

        dest = output_dir / filename
        url = entry["links"]["self"]

        if dest.exists() and not args.force:
            print(
                f"  EXISTS {filename} ({human_bytes(dest.stat().st_size)}; "
                "use --force to overwrite)"
            )
            skipped_existing += 1
            continue

        size_str = human_bytes(entry.get("size", 0))
        print(f"  GET   {filename} ({size_str})...", end="", flush=True)
        bytes_written = download_file(url, dest)
        print(f" done ({human_bytes(bytes_written)})")
        total_downloaded += bytes_written

    print()
    print(f"Downloaded {human_bytes(total_downloaded)} into {output_dir}")
    if skipped_existing:
        print(f"({skipped_existing} files already existed; use --force to re-download)")
    if skipped_zip:
        print("(Skipped fair_calibrate.zip; pass --include-zip if you need it)")

    print()
    print("Point the FaIRv2 adapter at the bundle with:")
    print()
    print(f"    export FAIR2_CALIBRATION_PATH={output_dir.resolve()}")
    print()
    print("Then run the end-to-end demo:")
    print()
    print("    python scripts/demo_fair2.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
