from __future__ import annotations

import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = PROJECT_ROOT / "DepthGuard_Submission.zip"

INCLUDE_DIRS = ["data", "docs", "reports", "scripts", "src", "tests", ".github"]
INCLUDE_FILES = [
    ".gitignore",
    "Dockerfile",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-vision.txt",
]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv"}


def should_include(path: Path) -> bool:
    return not any(part in SKIP_PARTS for part in path.parts)


def main() -> int:
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name in INCLUDE_FILES:
            file_path = PROJECT_ROOT / file_name
            if file_path.exists():
                archive.write(file_path, file_path.relative_to(PROJECT_ROOT))

        for directory_name in INCLUDE_DIRS:
            directory = PROJECT_ROOT / directory_name
            if not directory.exists():
                continue
            for file_path in directory.rglob("*"):
                relative_path = file_path.relative_to(PROJECT_ROOT)
                if file_path.is_file() and should_include(relative_path):
                    archive.write(file_path, relative_path)

    print(f"Created submission package: {OUTPUT_ZIP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
