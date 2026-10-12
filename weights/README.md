# Checkpoints

Model files are intentionally excluded from the source archive and ignored by Git.
No public checkpoint download URL has been supplied for this release.

For an existing local checkout, copy `Generator_iteration_199000.pth` from the
original `version3/weight/` directory into this folder. Its 747 state entries and
tensor shapes were checked against the released default network (25,114,700
parameters). This is a compatibility check, not a verification of manuscript scores.

Example:

```bash
python test.py --data-root datasets/All-in-One --weights weights/Generator_iteration_199000.pth
```

New training runs save plain model state dictionaries and `training_state.pt`
under `runs/AllFreqNet/checkpoints/`. Use `--weights` for initialization and
`--resume` for optimizer/scheduler continuation. Load files only from trusted sources.

For public distribution, upload selected model files as a GitHub Release asset
or to an appropriate model host, then add the actual download link and checksum here.
