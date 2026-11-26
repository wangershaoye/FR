# Automated facial-expression robustness analysis

## Purpose

This analysis tests whether the main associations remain after accounting for
visible facial expression in the stimuli. It is a robustness analysis, not a
claim that automated scores provide definitive expression labels.

## Feature extraction

The released features were generated with Py-Feat 0.6.2 (Cheong et al., 2023)
from the 500 stimulus images. The pipeline used RetinaFace face detection,
MobileFaceNet landmarks, the XGBoost AU models, and ResMaskNet expression
probabilities. Images were processed as 512-pixel padded inputs. For the two
images with a secondary detection, the highest-confidence face was retained.

Reference: Cheong, J. H., Jolly, E., Xie, T., Byrne, S., Kenney, M., and Chang,
L. J. (2023). Py-Feat: Python Facial Expression Analysis Toolbox. *Affective
Science, 4*, 781-796. https://doi.org/10.1007/s42761-023-00191-4

## Analysis decisions

- Primary expression covariate: AU12, the lip-corner puller.
- Sensitivity covariates: AU6 and ResMaskNet happiness probability.
- Features are standardized across the 500 stimuli before model fitting.
- Primary inference: Gaussian GEE with exchangeable working correlation and
  participant-clustered robust standard errors.
- Crossed-dependence check: maximum-likelihood mixed models with participant and
  stimulus random intercepts.
- The participant random intercept is explicitly requested with
  `re_formula="1"`; stimulus is specified as a variance component.

The GEE sensitivity models add one expression variable at a time. The
cross-classified models compare the baseline models with AU12-adjusted models.
This avoids entering the highly correlated AU6, AU12, and happiness scores in a
single model.

## Reproduce the statistical adjustment

The derived feature table is included and works with the base environment:

```bash
python src/expression_analysis.py
```

The four cross-classified fits are slower. For a fast GEE-only run:

```bash
python src/expression_analysis.py --skip-lmm
```

During a full run, the script writes per-model LMM checkpoints under the ignored
`.cache/expression_analysis/` directory.

## Regenerate image-derived features

Feature regeneration is optional and uses an isolated pinned environment
because Py-Feat 0.6.2 requires an older NumPy/XGBoost stack than the main
analysis environment.

```bash
python3 -m venv expression-venv
source expression-venv/bin/activate
python -m pip install -r requirements-expression.txt
python src/extract_expression_features.py
```

Py-Feat downloads pretrained resources on first use. Checkpoints and downloaded
resources under `.cache/` are excluded from version control. The extractor
overwrites `data/processed/stimulus_expression_features.csv` only after all 500
stimuli have valid AU12 scores.

## Interpretation limits

The pretrained expression models were developed primarily from photographs of
real faces and may not transfer perfectly to AI-synthesized faces. The scores
should therefore be described as automated model estimates. The adjustment can
show that the reported regression pattern is not explained by this measured
expression proxy, but it cannot rule out every aspect of facial expression.
