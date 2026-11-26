# AI-synthesized face stimuli

This directory contains the 500 full-color AI-synthesized face images used in
the experiment. Each image is a 1,024 x 1,024 PNG generated with StyleGAN2 using
an FFHQ-trained model. The images do not depict intentionally identified real
individuals.

The authoritative list is `../data/metadata/stimulus_manifest.csv`, which maps
each anonymous `stimulus_id` in the ratings data to its PNG filename and records
the file size, dimensions, and SHA-256 checksum.

The original stimulus directory contained 519 PNG files. Only the 500 filenames
referenced by the trial-level experimental data are included here; the 19 unused
files were excluded.

Validate the files from the repository root with:

```bash
python src/validate_stimuli.py
```

These binary files are configured for Git LFS. Run `git lfs install` before
adding them to a Git repository.

## License and attribution

These images are licensed under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0):
**non-commercial use only**, with attribution and share-alike. They were
generated with StyleGAN2 (NVIDIA) using an FFHQ-trained model, and the
non-commercial terms of StyleGAN2 and the FFHQ dataset flow through to these
synthetic images. Full terms, provenance, and credits are in
`../LICENSE-STIMULI.txt`; the repository-wide overview is in `../LICENSES.md`.

