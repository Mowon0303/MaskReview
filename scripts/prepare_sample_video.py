from __future__ import annotations

from pathlib import Path


def main() -> None:
    sample_dir = Path(__file__).resolve().parents[1] / "data" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"Put small mp4 samples in: {sample_dir}")


if __name__ == "__main__":
    main()

