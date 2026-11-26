# Data provenance

The analysis dataset was derived from 200 PsychoPy CSV files representing two
sessions for each of 100 participants. Only rows associated with an experimental
stimulus were retained. The three Likert responses and total trial response time
were joined to the analysis table using participant, session, and stimulus.

For the release candidate:

- Internal participant codes were replaced with deterministic anonymous IDs
  `P001` through `P100` based on sorted internal codes.
- Stimuli were replaced with `S001` through `S500` based on sorted generated
  filenames.
- The generated image basename was retained only in the stimulus manifest.
- Dates, raw mouse traces, PsychoPy participant entries, frame-rate metadata,
  and per-participant demographic rows were excluded.
- Aggregate participant demographics were retained in `sample_summary.csv`.
- No rating transformation, centering, winsorization, or imputation was applied.

The release contains 50,000 complete observations. Every participant has 500
trials and every stimulus has 100 ratings.

The `stimuli/images/` directory contains the 500 PNG files referenced by the
trial-level data. All files are 1,024 x 1,024 pixels. The release selection was
made from the original PsychoPy stimulus directory using the filenames in the
analysis data; 19 PNG files present in that directory but not used in the
experiment were excluded. Source and release SHA-256 values were verified to be
identical.

## Automated expression features

The 500 released stimulus images were processed with Py-Feat 0.6.2 using
RetinaFace for face detection, MobileFaceNet landmarks, XGBoost action-unit
models, and ResMaskNet expression probabilities. Images were resized with
aspect-ratio preservation and padding to 512-pixel model inputs. When more than
one face box was returned, the highest-confidence box was selected.

All 500 stimuli produced a selected face. Of these, 498 had exactly one
detected face; `S046` and `S241` had a low-confidence secondary box in the
background, while their highest-confidence central-face detections were
retained. The minimum selected-face confidence was 0.9921 and the median was
0.9983. The derived values are stored in
`data/processed/stimulus_expression_features.csv`. No pretrained model weights
are redistributed in this repository.

