# Uploading the source repository

Create an empty GitHub repository, then run these commands from this folder.
The repository URL for this release is:

```bash
git init
git add .
git status --short
git commit -m "Release cleaned AllFreqNet research code"
git branch -M main
git remote add origin https://github.com/llu362107-sudo/AllFreqNet.git
git push -u origin main
```

Git uses your local identity and authentication settings. The source archive
does not contain credentials or an existing Git history. If using GitHub's web
upload, upload this folder's contents, including `.gitignore` and `.gitattributes`.

Dataset arrays, checkpoints, local environment files and default result folders
are ignored. For pretrained models, publish selected files as Release assets
or through a model host, then update `weights/README.md` with a real download
link. Before describing the repository as a complete paper reproduction,
resolve the implementation differences and supply exact split/preprocessing
metadata described in `docs/IMPLEMENTATION_NOTES.md` and `docs/DATA.md`.
