"""Smoke tests for GAIA parquet generation.

Run with:
    python -m tests.smoke_gaia_data
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def _has_parquet_engine() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        return False


def test_make_gaia_data_text_only() -> bool:
    if not _has_parquet_engine():
        print("SKIP test_make_gaia_data_text_only (missing pyarrow/fastparquet)")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "gaia.jsonl"
        out_dir = root / "out"
        rows = [
            {
                "task_id": "text-1",
                "Question": "What is the capital of France?",
                "Final answer": "Paris",
                "Level": 1,
                "file_name": "",
            },
            {
                "task_id": "file-1",
                "Question": "Read the attached spreadsheet.",
                "Final answer": "42",
                "Level": 2,
                "file_name": "sheet.xlsx",
            },
        ]
        with src.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        subprocess.run(
            [
                sys.executable,
                "scripts/make_gaia_data.py",
                "--input-jsonl",
                str(src),
                "--out-dir",
                str(out_dir),
                "--out-prefix",
                "gaia_smoke",
            ],
            check=True,
        )

        expected = {
            "gaia_smoke.parquet": "search",
            "gaia_smoke_branch.parquet": "search_branch",
            "gaia_smoke_graph.parquet": "search_graph",
        }
        for name, workflow in expected.items():
            df = pd.read_parquet(out_dir / name)
            assert len(df) == 1
            row = df.iloc[0]
            assert row["ability"] == "GAIA"
            assert row["extra_info"]["workflow"] == workflow
            assert row["extra_info"]["answer"] == "Paris"
            assert row["extra_info"]["task_id"] == "text-1"
    return True


def main() -> None:
    if test_make_gaia_data_text_only():
        print("PASS test_make_gaia_data_text_only")


if __name__ == "__main__":
    main()
