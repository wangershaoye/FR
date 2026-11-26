# Public release checklist

## Blocking checks

- [x] Confirm that participant consent and ethics approval permit public release
      of de-identified trial-level ratings. The author confirmed this on
      2026-11-08.
- [x] Confirm that the generated face images may be redistributed. The author
      confirmed redistribution permission on 2026-11-08; final licence wording
      and attribution notices remain to be selected.
- [ ] Choose a software licence for `src/` and a data licence for `data/`.
- [ ] Add the complete author list, affiliations, ORCID identifiers, and contact.
- [ ] Review every file for direct identifiers, dates, free text, and hidden metadata.
- [ ] Run `python src/validate_data.py` in a clean environment.
- [ ] Run `python src/validate_stimuli.py` after cloning with Git LFS enabled.
- [ ] Run `python src/run_all.py` and compare generated tables with the manuscript.

## Repository checks

- [ ] Create a public or reviewer-accessible GitHub repository.
- [ ] Install Git LFS before adding the 500 PNG stimulus files.
- [ ] Create a tagged release, for example `v1.0.0`.
- [ ] Archive the tagged release in Zenodo and obtain a DOI.
- [ ] Add the DOI badge and preferred citation to `README.md`.
- [ ] Replace placeholders in `docs/DATA_AVAILABILITY.md`.
- [ ] Test the repository from a fresh clone outside the authors' machines.

## Revision-stage checks

- [x] Add automated facial-expression features and adjusted analyses for the revision.
- [ ] Add apparent-age coding only if collected and reported.
- [ ] Regenerate outputs after any change to the analysis dataset.
- [ ] Ensure the manuscript, response letter, repository, and archived release
      report the same sample sizes and model estimates.

