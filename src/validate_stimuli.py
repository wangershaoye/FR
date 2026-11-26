import hashlib

import pandas as pd
from PIL import Image

from common import ROOT


MANIFEST_PATH = ROOT / "data" / "metadata" / "stimulus_manifest.csv"
STIMULUS_DIR = ROOT / "stimuli" / "images"


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    required = {
        "stimulus_id",
        "generated_filename",
        "sha256",
        "size_bytes",
        "width_px",
        "height_px",
    }
    missing_columns = required.difference(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
    if len(manifest) != 500 or manifest["stimulus_id"].nunique() != 500:
        raise ValueError("The stimulus manifest must contain 500 unique stimuli.")

    expected = set(manifest["generated_filename"])
    actual = {
        path.name
        for path in STIMULUS_DIR.glob("*.png")
        if not path.name.startswith("._")
    }
    if expected != actual:
        raise ValueError(
            f"Stimulus file mismatch. Missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )

    for row in manifest.itertuples(index=False):
        path = STIMULUS_DIR / row.generated_filename
        if path.stat().st_size != int(row.size_bytes):
            raise ValueError(f"Size mismatch: {row.generated_filename}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row.sha256:
            raise ValueError(f"SHA-256 mismatch: {row.generated_filename}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.size != (int(row.width_px), int(row.height_px)):
                raise ValueError(f"Dimension mismatch: {row.generated_filename}")

    total_mib = manifest["size_bytes"].sum() / 1024**2
    print(f"Validated {len(manifest)} stimulus PNG files ({total_mib:.1f} MiB).")
    print("Dimensions: 1024 x 1024 pixels")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

