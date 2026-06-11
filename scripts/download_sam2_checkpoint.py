from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


CHECKPOINT_URLS = {
    "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a SAM2.1 checkpoint.")
    parser.add_argument("--model", choices=CHECKPOINT_URLS.keys(), default="tiny")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = CHECKPOINT_URLS[args.model]
    output_path = args.output_dir / Path(url).name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Already exists: {output_path}")
        return

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

