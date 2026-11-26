# Code repositories of Perceived Artificiality Markedly Reduces Trustworthiness but Only Minimally Affects Attractiveness in AI-Synthetic Faces 

## Contents

```text
data/processed/trial_level_ratings.csv   De-identified trial-level ratings
data/processed/stimulus_expression_features.csv  Automated expression features
data/metadata/stimulus_manifest.csv      Stimulus ID, filename, size, and SHA-256
data/metadata/sample_summary.csv         Aggregate participant description
data/metadata/data_dictionary.csv        Variable definitions
stimuli/images/                           The 500 AI-synthesized face stimuli
stimuli/README.md                         Stimulus description and validation
src/                                     Reproducible analysis scripts
notebooks/reproduce_all_analyses.ipynb   Notebook front end for the scripts
notebooks/figures.ipynb                   The manuscript figures, rendered inline
outputs/tables/                           Generated statistical tables
outputs/figures/                          Generated non-stimulus figures
```

## Licensing

This repository is released under three licenses, one per kind of material:

- Analysis code (`src/`, notebook code) — MIT (`LICENSE`).
- Author-generated data and results (`data/`, `outputs/`) — CC BY 4.0
  (`LICENSE-DATA.txt`).
- AI-synthesized face stimuli (`stimuli/images/`, and figures that reproduce
  them) — CC BY-NC-SA 4.0, non-commercial (`LICENSE-STIMULI.txt`).

The face images inherit the non-commercial StyleGAN2/FFHQ terms, so they may not
be used commercially even though the code and data are more permissive. GitHub
labels the repository MIT (from `LICENSE`); that label covers the code only. See
`LICENSES.md` for the full breakdown and the pre-publication checklist.




