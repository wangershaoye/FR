# Cross-classified model audit

## Finding

The earlier analysis notebook called `statsmodels.formula.api.mixedlm` with a
participant grouping variable and a stimulus variance-component formula, but
did not explicitly set `re_formula="1"`.

With statsmodels 0.14.0, this construction has:

```text
k_re = 0
participant random-effect design = None
k_vc = 1
```

Therefore, the earlier model included the stimulus variance component but did
not include a participant random intercept, despite the participant variable
being passed through `groups`.

## Release implementation

The release script uses:

```python
smf.mixedlm(
    formula,
    data=data,
    groups=data["participant_id"],
    re_formula="1",
    vc_formula={"stimulus": "0 + C(stimulus_id)"},
)
```

This explicitly estimates both the participant random-intercept variance and
the stimulus variance component. The corrected model preserves the substantive
pattern: artificiality is negatively associated with perceived trustworthiness,
whereas its adjusted association with attractiveness is small and positive.

## Action before manuscript resubmission

Use the corrected estimates in the revised manuscript and response letter.
Do not describe the earlier saved output as a participant-plus-stimulus
cross-classified model. Re-run the corrected models again after adding any new
facial-expression covariates.

