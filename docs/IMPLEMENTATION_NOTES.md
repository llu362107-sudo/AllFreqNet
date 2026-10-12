# Implementation and reproducibility notes

This release was prepared from the supplied `version3` folder and checked
against `Revised manuscript_R2.pdf`. It preserves the current network instead
of reconstructing a different model from the paper diagram.

## Paper-to-code map

| Concept | Implementation |
| --- | --- |
| Global task descriptor / auxiliary classifier | `GlobalTaskDescriptor` in `model/task_descriptor.py` |
| Multi-scale expert routing | `MSMoE`, `TaskAdaptiveMoE`, `SparseExpertDispatcher` in `model/ms_moe.py` |
| Four local experts, top-2 selection, global branch | `TaskAdaptiveMoE` |
| Spatially varying grouped low-pass kernels | `RegionAdaptiveFrequencyDecoupling` in `model/rc_fpm.py` |
| Coordinate-conditioned frequency prompts | `CoordinateConditionedFrequencyPromptGenerator`, `RCFPM` in `model/rc_fpm.py` |
| Four-level restoration backbone | `AllFreqNet` in `model/allfreqnet.py` |
| L1 + 0.001 CE + 0.001 balancing | `train.py` |

The default network has 25,114,700 parameters and 747 state dictionary entries.
Defaults include width 42, block counts [5,7,7,9], four refinement blocks and
attention head counts [1,2,4,8]. Parameter names are preserved for strict loading
of existing checkpoints. Formatting and the removal of an unused SymPy import
do not change tensor operations. Classes and source files were subsequently
renamed to align with the manuscript; [NAMING.md](NAMING.md) records the mapping.
The frequency decoupling and prompt modulation stages remain separately
registered so the original checkpoint hierarchy is retained.

## Preserved behaviors that merit author review

1. **First encoder routing output.** In `AllFreqNet.forward`,
   `inp_enc_level1_moe` is computed, but `encoder_level1` receives the original
   `inp_enc_level1`. This branch contributes its balancing loss during training,
   while its restored features are unused. The paper describes MS-MoE at each
   encoder stage. This release preserves the implementation, including the
   unused routed tensor, to keep existing checkpoint behavior.
2. **Expert merging.** `SparseExpertDispatcher.combine` exponentiates local expert
   outputs, computes their gated sum, then takes a logarithm. The paper's
   equation writes a direct weighted sum. Replacing the current log-domain
   merge would change predictions without changing checkpoint shapes.
3. **Descriptor and diagram details.** `GlobalTaskDescriptor.forward` averages CNN features and
   returns a 32-dimensional descriptor directly; it does not apply a final
   descriptor softmax. The default first encoder has five Transformer blocks;
   the supplied architecture figure labels the first stage with three. A
   checkpoint's strict compatibility cannot by itself resolve these differences.

These are recorded facts, not corrections to the architecture. Confirm which
implementation produced the reported manuscript results before claiming exact
paper reproduction or changing these operations.

## Changes in the release wrapper

- Removed local drive/server paths, forced GPU visibility and import-time runs.
- Replaced per-case scripts with parameterized training, test and inference.
- Validation now uses all available samples by default; the original training
  entry passed `use_num=0`, producing no validation samples.
- Per-modality validation limits are applied independently. The original loader
  reused the first modality's count for later modalities.
- Unified NumPy/tensor conversion to `[1,H,W]` for all formats/modalities; this
  avoids extra axes for RAW files and concatenating non-tensor BIN arrays.
- Replaced whole-string `LQ`→`HQ` substitution with sibling-folder pairing;
  added explicit errors for missing pairs, empty datasets and duplicate names.
- RAW reading no longer applies CT-specific clipping before the modality's own
  transform. The old reader clipped every RAW image, including MRI, to CT bounds.
- Retained intensity and evaluation ranges, L1/CE/MoE weights, optimizer and
  learning-rate defaults. Seeding is now enabled by default (`--seed 42`); it was
  commented out in the original entry point.
- Model saving no longer moves the live model to CPU and back. Resume restores
  optimizer/scheduler state but is not bitwise continuation of data sampling.
- Optional inference padding supports odd sizes without changing the network.

The wrapper fixes can change data coverage, selected best checkpoints and future
training trajectories. The guarantee of identical outputs applies to the same
existing weights and the same model input, not to retraining with corrected data
handling.

## Metric convention

Evaluation runs with batch size 1, denormalizes prediction/target, then applies
the modality's evaluation clipping. Per-slice `data_range` is `target.max() -
target.min()`. SSIM uses an 11×11 Gaussian window (sigma 1.5) and zero padding,
matching the original implementation for ordinary single-channel 4D tensors.
Task means average per-slice metrics; overall means average all slices together.
This is not the same as averaging per-task means with equal task weights.
Constant targets return NaN for PSNR/SSIM, with valid counts reported. Identical
nonconstant images yield infinite PSNR; JSON serializes non-finite values as null.

## Verification scope

- Strictly loaded the original `Generator_iteration_199000.pth`.
- Compared original and cleaned network outputs on a synthetic 1×1×16×24 input
  using that checkpoint: bitwise equal on CPU.
- Passed nine regression checks covering shapes, pairing, per-modality counts,
  synchronized crops, NIfTI metadata, normalization, rectangular metrics and
  the default network's forward/backward/padding contracts.
- Completed two synthetic training iterations, validation across four tasks,
  paired evaluation and LQ-only inference/export. No real patient data was used.
- Checked synthetic single-frame DICOM export (including geometry and UID
  consistency) and continuation from an interrupted optimizer/scheduler checkpoint.
- After splitting/renaming the model files, strictly loaded the original weights
  again; inference outputs, training outputs and parameter gradients were bitwise
  equal to the original model with matched inputs and random seeds. All registered
  checkpoint keys and tensor values were preserved.

This cleanup does not establish the original dataset split, reproduce the full
200,000-iteration training, verify CUDA behavior, or validate paper-table scores.
The revision's organ-boundary/reader-study scripts were case-specific and are
not packaged as generic reproduction tools: they include dataset-specific mask
orientation assumptions, and the required images/masks/readings were not supplied.
