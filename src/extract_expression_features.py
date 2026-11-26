import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import EXPRESSION_FEATURE_PATH, ROOT


IMAGE_DIR = ROOT / "stimuli" / "images"
MANIFEST_PATH = ROOT / "data" / "metadata" / "stimulus_manifest.csv"
CACHE_DIR = ROOT / ".cache" / "expression_features"
CHECKPOINT_PATH = CACHE_DIR / "pyfeat_checkpoint.csv"
SUMMARY_PATH = CACHE_DIR / "pyfeat_extraction_summary.json"


def best_face_index(face_rows):
    if len(face_rows) == 0:
        return None
    return int(np.argmax([float(face[-1]) for face in face_rows]))


def build_detector():
    try:
        from feat import Detector
    except ImportError as error:
        raise RuntimeError(
            "Py-Feat is not installed. Install requirements-expression.txt in a "
            "separate environment before regenerating image features."
        ) from error

    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        facepose_model="img2pose",
        identity_model="facenet",
        device="cpu",
        n_jobs=1,
        verbose=False,
    )
    detector.au_model.au_keys = ["AU6", "AU12"]
    return detector


def process_batch(detector, batch_manifest, output_size):
    from feat.data import ImageDataset
    from feat.utils import FEAT_EMOTION_COLUMNS
    from torch.utils.data import DataLoader

    paths = [str(IMAGE_DIR / name) for name in batch_manifest["generated_filename"]]
    loader = DataLoader(
        ImageDataset(
            paths,
            output_size=output_size,
            preserve_aspect_ratio=True,
            padding=True,
        ),
        batch_size=len(paths),
        num_workers=0,
        shuffle=False,
    )
    images = next(iter(loader))["Image"]
    faces = detector.detect_faces(images, threshold=0.5)
    landmarks = detector.detect_landmarks(images, faces)
    aus = detector.detect_aus(images, landmarks)
    emotions = detector.detect_emotions(images, faces, landmarks)

    rows = []
    for image_index, manifest_row in batch_manifest.reset_index(drop=True).iterrows():
        face_rows = faces[image_index]
        selected = best_face_index(face_rows)
        row = {
            "stimulus_id": manifest_row["stimulus_id"],
            "generated_filename": manifest_row["generated_filename"],
            "n_faces_detected": len(face_rows),
            "selected_face_index": selected,
            "face_confidence": np.nan,
            "AU6": np.nan,
            "AU12": np.nan,
            **{name: np.nan for name in FEAT_EMOTION_COLUMNS},
        }
        if selected is not None:
            row["face_confidence"] = float(face_rows[selected][-1])
            row["AU6"] = float(aus[image_index][selected, 0])
            row["AU12"] = float(aus[image_index][selected, 1])
            for emotion_index, emotion_name in enumerate(FEAT_EMOTION_COLUMNS):
                row[emotion_name] = float(
                    emotions[image_index][selected, emotion_index]
                )
        rows.append(row)
    return pd.DataFrame(rows)


def run(batch_size=25, output_size=512, force=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EXPRESSION_FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)[["stimulus_id", "generated_filename"]]
    manifest = manifest.sort_values("stimulus_id").reset_index(drop=True)
    if len(manifest) != 500 or manifest["stimulus_id"].nunique() != 500:
        raise ValueError("Stimulus manifest must contain 500 unique stimuli.")

    if force:
        completed = pd.DataFrame()
    elif CHECKPOINT_PATH.exists():
        completed = pd.read_csv(CHECKPOINT_PATH)
        print(f"Resuming from {len(completed)} checkpointed stimuli.")
    else:
        completed = pd.DataFrame()
    done_ids = set(completed.get("stimulus_id", []))
    pending = manifest.loc[~manifest["stimulus_id"].isin(done_ids)].copy()

    detector_start = time.perf_counter()
    detector = build_detector()
    detector_seconds = time.perf_counter() - detector_start
    batch_seconds = []
    for start in range(0, len(pending), batch_size):
        batch = pending.iloc[start : start + batch_size]
        batch_start = time.perf_counter()
        output = process_batch(detector, batch, output_size)
        elapsed = time.perf_counter() - batch_start
        batch_seconds.append(elapsed)
        completed = pd.concat([completed, output], ignore_index=True)
        completed = completed.drop_duplicates("stimulus_id", keep="last")
        completed.to_csv(CHECKPOINT_PATH, index=False)
        print(f"Processed {len(completed)}/{len(manifest)}; batch seconds={elapsed:.2f}")

    final = manifest.merge(
        completed, on=["stimulus_id", "generated_filename"], how="left"
    )
    if final["AU12"].notna().sum() != len(manifest):
        missing = final.loc[final["AU12"].isna(), "stimulus_id"].tolist()
        raise RuntimeError(f"Expression features are missing for: {missing}")
    final.to_csv(EXPRESSION_FEATURE_PATH, index=False)

    summary = {
        "n_requested": len(manifest),
        "n_with_face": int((final["n_faces_detected"] > 0).sum()),
        "n_exactly_one_face": int((final["n_faces_detected"] == 1).sum()),
        "n_multiple_faces": int((final["n_faces_detected"] > 1).sum()),
        "face_confidence_min": float(final["face_confidence"].min()),
        "face_confidence_median": float(final["face_confidence"].median()),
        "detector_initialization_seconds": detector_seconds,
        "batch_size": batch_size,
        "output_size": output_size,
        "batch_seconds": batch_seconds,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {EXPRESSION_FEATURE_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate Py-Feat expression features from the 500 stimuli."
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--force", action="store_true", help="Ignore any checkpoint.")
    args = parser.parse_args()
    run(args.batch_size, args.output_size, args.force)


if __name__ == "__main__":
    main()
