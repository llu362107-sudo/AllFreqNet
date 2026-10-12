# Data conventions

## Inputs

Each file represents one aligned single-channel 2D image. Arrays shaped `[H,W]`,
`[1,H,W]` or with other singleton dimensions are accepted; a volume with several
slices is rejected. The model restores the input spatial size, so MRI LQ inputs
must already be on the HQ spatial grid. Volume conversion, registration and
k-space degradation are not performed by the training loader.

| Extension | Interpretation |
| --- | --- |
| `.npy` | Numeric array, loaded with `allow_pickle=False` |
| `.nii`, `.nii.gz` | SimpleITK intensity array; single slice only |
| `.raw` | Little-endian float32, default shape 512 × 512 |
| `.bin` | Original version3 pickled array/tensor; trusted local files only |
| `.dcm` | pydicom pixel array with RescaleSlope/RescaleIntercept applied |

All inputs are converted to float32 `[1,H,W]`. Files are sorted, pairing is
validated, and duplicate output stems within a modality/split are rejected.
Training uses the same random crop for LQ and HQ. No new geometric augmentation
or sampling rebalancing was added; shuffling is over the concatenated slices.

## Intensities

Provide images in these physical/scanner units, not images already normalized
to [0,1]. Normalization clips first, then linearly maps to [0,1].

| Directory | Task | Input/denormalization range | Evaluation/output clipping |
| --- | --- | --- | --- |
| `CT` | CT-DN | [-1024, 3071] HU | [-1000, 1000] HU |
| `MA` | CT-MAR | [-1024, 3071] HU | [-1000, 1000] HU |
| `PET` | PET-Syn | [0, 20] | [0, 20] |
| `MRI` | MR-SR | [0, 4095] | [0, 4095] |

The auxiliary classification labels are fixed: PET=0, CT=1, MRI=2, MA=3,
regardless of the order passed to `--modalities`. For PET, verify that your
preprocessing and intensity scaling match the original experiment.

## Split information from the R2 manuscript

| Task | Train | Validation | Test |
| --- | --- | --- | --- |
| CT-DN | 8 cases | 1 case | 1 case |
| CT-MAR | 11,200 slices | 1,400 slices | 1,400 slices |
| PET-Syn | 120 cases | 10 cases | 29 cases |
| MR-SR | 405 volumes | 58 volumes | 115 volumes |

The manuscript describes the 2016 NIH-AAPM/Mayo low-dose CT benchmark, an AAPM
CT-MAR dataset, a paired PET dataset, and IXI T2 MRI. MRI preprocessing retains
the central 100 slices and crops k-space for 4× downsampling (central 6.25% of
points). The supplied code folder does not contain authoritative per-case split
manifests or a self-contained public preprocessing pipeline. Counts alone do
not identify the exact data split. Obtain the datasets and access permissions
from their owners and supply the original manifests to reproduce paper scores.

The old AMIR README's dataset link was not promoted to an AllFreqNet dataset
download: it does not establish the full four-task R2 dataset and split.

## Output geometry

NIfTI export copies spacing, origin and direction from a compatible NIfTI source,
including single-slice 3D files. RAW/BIN/NPY inputs have no spatial metadata;
their NIfTI output has SimpleITK's default geometry. Preserve or supply geometry
separately when assembling slices into volumes.

`infer_dicom.py` preserves source patient/study and geometry metadata and writes
derived CT images with new series/SOP UIDs, signed 16-bit pixels and explicit
little-endian transfer syntax. Compressed and multi-frame DICOM are rejected.
Source metadata is retained; this is not an anonymization utility.
