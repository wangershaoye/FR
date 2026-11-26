# GitHub publication guide

This directory is not initialized as a Git repository. The steps below keep the
first public history clean and ensure the 500 PNG stimuli are stored with Git
Large File Storage (Git LFS), not ordinary Git objects.

## 1. Complete the release blockers

Before publishing, complete `docs/RELEASE_CHECKLIST.md`. In particular:

- choose the code, data, and stimulus licences;
- add author/contact metadata;
- replace repository/DOI placeholders only after stable links exist;
- inspect all tracked files for identifiers, secrets, local paths, and hidden
  metadata.

A practical licence combination is MIT for analysis code and CC BY 4.0 for
author-generated tabular data. The stimulus images need their own notice that
matches the applicable StyleGAN/FFHQ source terms and the authors' redistribution
rights. Do not publish until that wording is confirmed.

## 2. Install Git LFS on macOS

```bash
brew install git-lfs
git lfs install
```

The repository already contains `.gitattributes`, which assigns
`stimuli/images/*.png` to Git LFS. GitHub's installation and configuration
guides are:

- https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage
- https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage

## 3. Create an empty repository on GitHub

1. Sign in to GitHub and select **New repository**.
2. Choose the repository name and description.
3. Prefer **Private** for the first push and final online inspection.
4. Do not initialize it with a README, `.gitignore`, or licence because those
   files already exist locally.
5. Create the repository and copy its HTTPS URL.

GitHub's official local-code import guide is:
https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github

## 4. Initialize and commit the local release

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` below.

```bash
cd /path/to/github_release
git init -b main
git lfs install
git add .
git status --short
git lfs ls-files | wc -l
git commit -m "Initial reproducible release"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git remote -v
git push -u origin main
```

`git lfs ls-files | wc -l` should report `500` before the first push. If it
reports zero, stop: otherwise the large images may enter ordinary Git history.

GitHub no longer accepts account passwords for HTTPS Git operations. Complete
the browser/device authentication prompt or use a personal access token when
requested.

## 5. Inspect and make the repository public

On GitHub, verify that:

- the README renders correctly;
- the data contain only anonymous participant IDs;
- all 500 image files display as Git LFS objects;
- no local cache, model weights, raw participant files, or credentials appear;
- licences and author metadata are complete.

Then open **Settings**, scroll to **Danger Zone**, select **Change repository
visibility**, and choose **Public**. GitHub documents the consequences and exact
steps here:
https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility

## 6. Tag the version and create a release

After the public contents are final:

```bash
git tag -a v1.0.0 -m "Initial public release"
git push origin v1.0.0
```

On the GitHub repository page, open **Releases**, choose **Draft a new release**,
select `v1.0.0`, add concise release notes, and publish it. The official guide is:
https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository

## Large-file bandwidth note

The stimuli total about 640 MiB. GitHub currently includes 10 GiB of Git LFS
storage and 10 GiB of monthly LFS bandwidth for personal Free and Pro plans,
but these quotas and billing rules can change. Confirm the current allowance at:
https://docs.github.com/en/billing/concepts/product-billing/git-lfs

GitHub source archives contain only LFS pointer files by default. Repository
administrators can enable **Settings > Archives > Include Git LFS objects in
archives**, but archive downloads then consume LFS bandwidth. For durable
citation and heavier public downloading, archive the tagged release and stimuli
in Zenodo and use the resulting DOI in `docs/DATA_AVAILABILITY.md`.
