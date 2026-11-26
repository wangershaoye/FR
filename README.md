# Perceived artificiality, attractiveness, and trustworthiness ratings

This repository contains the de-identified trial-level data and analysis code
for a study in which 100 participants rated 500 AI-synthesized faces on
artificiality, attractiveness, and trustworthiness using 5-point Likert scales.
Each participant rated every stimulus, yielding 50,000 observations.

## Repository status

This directory is a release candidate prepared for GitHub/Zenodo deposition.
Before public release, complete the checks in `docs/RELEASE_CHECKLIST.md`, add
author metadata, replace the placeholder copyright holder in `LICENSE`, and
confirm that participant consent and ethics approval permit public sharing of
de-identified ratings. See the Licensing section below.

The release includes exactly the 500 face images used in the experiment. The
stimulus manifest maps anonymous stimulus IDs to generated filenames and records
the size, dimensions, and SHA-256 checksum of every PNG.

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
docs/                                     Provenance and release documentation
```

`docs/OUTPUT_MAP.md` links every analysis script to its generated tables and
figures. `docs/GITHUB_RELEASE_GUIDE.md` gives the exact first-push, Git LFS,
visibility, and versioned-release workflow.

## Reproduce the analyses

Python 3.10 was used for the archived environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/validate_data.py
python src/validate_stimuli.py
python src/run_all.py
```

The automated expression features used in the reviewer-requested robustness
analysis are included, so reproducing the statistical adjustment does not
require rerunning the image models. The standard command above reproduces that
analysis with the base environment.

`notebooks/figures.ipynb` reproduces the manuscript figures inline from the
public data in the article's own visualization scheme, so they can be browsed
without running the full pipeline. `notebooks/reproduce_all_analyses.ipynb` runs
the complete set of analysis scripts end to end.

Alternatively, create the Conda environment:

```bash
conda env create -f environment.yml
conda activate ai-face-ratings
python src/run_all.py
```

The cross-classified mixed models use 50,000 observations. The complete run,
including the four expression robustness models, can take 20 minutes or longer
depending on the machine. To reproduce the faster analyses first:

```bash
python src/run_all.py --skip-lmm
```

To omit the expression robustness analysis entirely, add
`--skip-expression`. See `docs/EXPRESSION_ANALYSIS.md` for the separate,
optional environment needed to regenerate the image-derived features.

## Analysis sequence

1. Validate the complete 100 participant x 500 stimulus crossing.
2. Compute participant-wise zero-order Spearman correlations.
3. Compute participant-wise partial Spearman correlations.
4. Apply Fisher's z transformation and group-level one-sample tests.
5. Fit Gaussian GEE models with exchangeable and independent working correlations.
6. Fit cross-classified mixed models with participant and stimulus random intercepts.
7. Generate the correlation, partial-correlation, and GEE prediction figures.
8. Test the artificiality-rating relationships for non-linearity (uncanny-valley
   curvature) with participant-wise and stimulus-level quadratic models.
9. Refit the GEE and cross-classified models with automated facial-expression
   covariates and generate the facial-expression figures and tables.

## Main models

```text
Trustworthiness ~ Artificiality + Attractiveness
Attractiveness ~ Artificiality + Trustworthiness
```

GEE standard errors are robust/sandwich estimates clustered by participant.
The cross-classified models include random intercepts for participant and
stimulus.

## Non-linearity (uncanny-valley) test

`src/nonlinearity_analysis.py` tests whether perceived artificiality relates to
trustworthiness and attractiveness non-linearly, as an uncanny-valley dip would
predict, or monotonically. It fits participant-wise quadratic regressions,
stimulus-level nested quadratic models with an added-term F test and change in
R-squared, and reports decile trends and a least- versus most-artificial
quintile contrast. The analysis uses only the public trial-level ratings and the
base environment. See `docs/NONLINEARITY_ANALYSIS.md` for the interpretation.

## Facial-expression robustness analysis

Facial-expression features were extracted for all 500 stimuli with Py-Feat
0.6.2. Standardized AU12 (lip-corner puller) is the primary expression
covariate; standardized AU6 and the model's happiness probability are
sensitivity variables. The analysis code, derived feature table, model tables,
the AU12-versus-ratings figure, and the per-participant robustness figure are
included. Py-Feat model weights are not included
and are downloaded by Py-Feat only when the optional feature-extraction script
is run.

## Mixed-model implementation note

The release code explicitly sets `re_formula="1"` in `statsmodels.MixedLM`.
This is required for a participant random intercept when a stimulus
`vc_formula` is also supplied. See `docs/MODEL_AUDIT.md` for the audit of the
earlier notebook implementation. The release model matches the random-effects
structure stated in the manuscript methods.

## Data protection

The public participant identifiers (`P001`-`P100`) are newly assigned and do
not reproduce the internal study codes. Dates, computer interaction traces,
original PsychoPy participant entries, and individual demographic records are
not included. Only aggregate demographic information is provided.

## Large stimulus files

The 500 PNG files total approximately 640 MiB. They are configured for Git LFS
through `.gitattributes`. Install Git LFS and run `git lfs install` before the
first commit. For durable citation and unrestricted download bandwidth, archive
the tagged GitHub release in Zenodo and cite the Zenodo DOI.

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

## Citation and archive

After the GitHub release is final, create a versioned Zenodo archive and replace
the placeholders in `docs/DATA_AVAILABILITY.md` with the DOI. Add the final
article citation and author list before publication.


