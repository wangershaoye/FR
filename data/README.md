# Data files

## `processed/trial_level_ratings.csv`

Analysis-ready long-format ratings. Participant IDs were reassigned for this
release. Ratings are the original integer responses on 1-5 Likert scales.

## `processed/stimulus_expression_features.csv`

One row per stimulus containing Py-Feat 0.6.2 face-detection quality-control
fields, AU6 and AU12 scores, and seven expression-category probabilities. The
highest-confidence detected face was retained for each image. These are
model-derived stimulus descriptors, not human-coded expressions.

## `metadata/stimulus_manifest.csv`

Maps each anonymous stimulus ID to the generated seed filename used in the
experiment. The corresponding 500 image files are in `stimuli/images/`.

## `metadata/sample_summary.csv`

Aggregate sample characteristics only. Individual age and sex records are not
included because they are not required by the reported statistical models.

## `metadata/data_dictionary.csv`

Definitions, data types, ranges, and missing-value conventions.

## Missing values

There are no missing values in the released trial-level analysis variables.

