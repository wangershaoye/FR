# Non-linearity (uncanny-valley) test

## Purpose

This analysis tests whether perceived artificiality relates to trustworthiness
and attractiveness non-linearly, as an uncanny-valley dip would predict, or
whether the relationships are monotonic. It is a reviewer-requested robustness
check on the linear associations reported in the main analyses.

## Methods

Three complementary views are reported, all from the public trial-level ratings.

1. **Participant-wise quadratic regression.** For each participant, the rating is
   regressed on mean-centred artificiality and its square. The linear and
   quadratic coefficients are then tested against zero across participants with a
   one-sample t test, mirroring the per-participant logic used for the
   correlations.
2. **Stimulus-level aggregate regression.** Per-stimulus mean ratings are
   regressed on mean-centred stimulus-mean artificiality. The added quadratic
   term is assessed with a nested F test and its change in R-squared, and the
   fitted vertex is compared with the observed artificiality range to check
   whether any curvature implies an interior minimum within the measured range.
3. **Ordered-bin trend and extreme-group contrast.** Stimuli are grouped into
   deciles of mean artificiality to show the shape of the trend, and the
   least- and most-artificial quintiles are contrasted directly.

The analysis uses only numpy, pandas, scipy, and matplotlib, and requires no
image models or expression features.

## Reproduce

```bash
python src/nonlinearity_analysis.py
```

Outputs:

- `outputs/tables/nonlinearity_participantwise.csv`
- `outputs/tables/nonlinearity_stimulus_level.csv`
- `outputs/tables/nonlinearity_artificiality_deciles.csv`
- `outputs/figures/nonlinearity_artificiality_trends.png` / `.pdf`

## Results summary

**Trustworthiness.** The relationship is monotonic and negative. The quadratic
term is not significant either participant-wise (p = 0.50) or at the stimulus
level (added-quadratic F(1, 497) = 1.93, p = 0.17; change in R-squared about
0.003). The fitted vertex falls outside the observed artificiality range, so the
fit has no interior minimum. Decile means decline steadily (about 3.40 at the
least-artificial decile to about 2.47 at the most-artificial decile), and the
least-artificial quintile is rated far more trustworthy than the most-artificial
quintile (about 3.32 versus 2.59, p < .001).

**Attractiveness.** The relationship is essentially flat. The quadratic term is
not significant participant-wise (p = 0.25). At the stimulus level it is
statistically detectable (added-quadratic F(1, 497) = 6.77, p = 0.010) but
explains only about 1.3% of the variance, describing a very shallow upward
curvature rather than a pronounced valley. The least- and most-artificial
quintiles do not differ (about 3.00 versus 3.01, n.s.).

## Interpretation

The data span a wide range of perceived artificiality but show no uncanny-valley
dip: the trustworthiness decline is monotonic, and the attractiveness curvature
is negligible in magnitude. This is consistent with modern GAN-synthesized faces
sitting on the far, human-like side of the uncanny-valley continuum rather than
within the valley, so a monotonic (linear) description is adequate for these
stimuli. The exact test statistics are written to the tables above; the figure
shows the stimulus-level scatter with the linear and quadratic fits and the
quintile means.
