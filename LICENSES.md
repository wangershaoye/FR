# Licensing

This repository combines three kinds of material under three separate licenses.

| Material | Location | License | File |
|---|---|---|---|
| Analysis code | `src/`, notebook code in `notebooks/`, config files | **MIT** | `LICENSE` |
| Author-generated data and results | `data/`, `outputs/tables/`, `outputs/figures/` | **CC BY 4.0** | `LICENSE-DATA.txt` |
| AI-synthesized face stimuli | `stimuli/images/`, and figures that reproduce them | **CC BY-NC-SA 4.0** (non-commercial) | `LICENSE-STIMULI.txt` |

Key points:

- The code is permissively licensed (MIT).
- The data and results may be reused, including commercially, with attribution
  (CC BY 4.0).
- The face images are **non-commercial only** (CC BY-NC-SA 4.0), matching the
  upstream StyleGAN2 and FFHQ terms. They may not be used commercially.
- Where author-generated materials incorporate the stimulus images (for example
  the example-face panels in `notebooks/figures.ipynb`), the stimulus license
  (CC BY-NC-SA 4.0) also applies to those portions.

GitHub reports the repository license as MIT because `LICENSE` is the MIT text;
that label covers the code only. The data and stimulus licenses above apply to
their respective directories as stated.

## Before non-anonymous publication

- Replace the placeholder copyright holder "The Author(s)" in `LICENSE`, and the
  bracketed citation placeholders in `LICENSE-DATA.txt`, with the real
  author/copyright information.
- Confirm that the specific StyleGAN2 model/weights used permit redistribution
  of the selected generated images under CC BY-NC-SA 4.0.
- Confirm that participant consent and institutional ethics approval permit
  public release of the de-identified trial-level ratings.

Do not treat the absence of a license on any individual file as permission to
reuse it outside these terms.
