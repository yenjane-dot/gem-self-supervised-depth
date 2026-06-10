"""
Scan the public GEM repository for common accidental-publication artifacts.

Run:
    python baselines/monodepth2/scan_public_repo.py
"""

from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".idea",
    ".vscode",
    "outputs",
    "logs",
    "checkpoints",
    "tmp",
    "temp",
}

SUSPICIOUS_SUFFIXES = {
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
}

SUSPICIOUS_NAME_PARTS = [
    "cover_letter",
    "submission",
    "manuscript",
    "neurocomputing",
    "authorhub",
    "reviewer",
    "response_to_reviewer",
    "technical_check",
]


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def main() -> None:
    findings: list[Path] = []

    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or should_skip(path):
            continue

        lower_name = path.name.lower()
        lower_rel = str(path.relative_to(REPO_ROOT)).lower()

        if path.suffix.lower() in SUSPICIOUS_SUFFIXES:
            findings.append(path)
            continue

        if any(token in lower_name or token in lower_rel for token in SUSPICIOUS_NAME_PARTS):
            findings.append(path)

    print("=" * 72)
    print("Public repository safety scan")
    print("=" * 72)
    if not findings:
        print("No suspicious publication artifacts detected.")
        return

    print("Review the following files before pushing:")
    for item in sorted(findings):
        print(f"- {item.relative_to(REPO_ROOT)}")

    print("=" * 72)
    print(f"Total suspicious files: {len(findings)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
