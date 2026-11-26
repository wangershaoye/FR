# Analysis and output map

| Script | Analysis | Generated tables | Generated figures |
|---|---|---|---|
| `src/validate_data.py` | Completeness, ranges, crossing structure | Console report | None |
| `src/validate_stimuli.py` | Stimulus count, names, dimensions, sizes, and SHA-256 | Console report | None |
| `src/correlations.py` | Participant-wise zero-order and partial Spearman correlations; Fisher z tests | `participant_zero_order_spearman.csv`, `participant_partial_spearman.csv`, `fisherz_zero_order_summary.csv`, `fisherz_partial_summary.csv` | None |
| `src/gee_analysis.py` | Exchangeable and independent Gaussian GEE models | `gee_coefficients.csv` | `gee_marginal_predictions.png/.pdf` |
| `src/mixed_models.py` | Participant-plus-stimulus cross-classified mixed models | `cross_classified_fixed_effects.csv`, `cross_classified_model_information.csv` | None |
| `src/figures.py` | Correlation and partial-correlation distributions | Uses participant-level tables above | `participant_spearman_distributions.png/.pdf`, `partial_correlation_joint_distribution.png/.pdf` |
| `src/nonlinearity_analysis.py` | Non-linearity (uncanny-valley) test: participant-wise quadratic regressions, stimulus-level nested quadratic F tests, and ordered-bin and extreme-quintile contrasts | `nonlinearity_participantwise.csv`, `nonlinearity_stimulus_level.csv`, `nonlinearity_artificiality_deciles.csv` | `nonlinearity_artificiality_trends.png/.pdf` |
| `src/expression_analysis.py` | AU12-adjusted GEE and cross-classified LMM; AU6 and happiness GEE sensitivity analyses; expression descriptives | `expression_feature_descriptives.csv`, `expression_rating_spearman_correlations.csv`, `expression_gee_fixed_effects.csv`, `expression_gee_model_information.csv`, `expression_gee_artificiality_comparison.csv`, `expression_lmm_fixed_effects.csv`, `expression_lmm_model_information.csv`, `expression_lmm_artificiality_comparison.csv` | `expression_au12_vs_ratings.png/.pdf`, `expression_partial_correlation_robustness.png/.pdf` |
| `src/extract_expression_features.py` | Optional Py-Feat regeneration from the 500 images | Replaces `data/processed/stimulus_expression_features.csv`; cache summary is ignored by Git | None |

Every non-image figure is drawn by `src/manuscript_figures.py`, the single
source of truth shared by the scripts above and by `notebooks/figures.ipynb`, so
the released figure files and the inline notebook figures are identical. The
release also contains all 500 face images.

