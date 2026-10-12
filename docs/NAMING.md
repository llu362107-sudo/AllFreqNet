# Paper-aligned model names

The public model is a real `AllFreqNet` class, defined in `model/allfreqnet.py`.
The previous `AMIR` alias and `Model_AMIR.py` source file have been replaced.

| Paper concept / function | Current class | File | Previous class |
| --- | --- | --- | --- |
| AllFreqNet | `AllFreqNet` | `model/allfreqnet.py` | `AMIR` |
| MS-MoE stage | `MSMoE` | `model/ms_moe.py` | `Spatial_Routing` |
| Task-adaptive local/global expert core | `TaskAdaptiveMoE` | `model/ms_moe.py` | `MoE` |
| Sparse token dispatch | `SparseExpertDispatcher` | `model/ms_moe.py` | `SparseDispatcher` |
| Expert MLP | `ExpertMLP` | `model/ms_moe.py` | `Mlp` |
| Region-adaptive frequency decoupling | `RegionAdaptiveFrequencyDecoupling` | `model/rc_fpm.py` | `RegionAdaptiveKernelFilter` |
| Coordinate-conditioned frequency prompt generation | `CoordinateConditionedFrequencyPromptGenerator` | `model/rc_fpm.py` | `FrequencySpecificPromptGenerator_Coord` |
| RC-FPM prompt modulation and fusion stage | `RCFPM` | `model/rc_fpm.py` | `PromptModule_Coord` |
| Global task descriptor and auxiliary classifier | `GlobalTaskDescriptor` | `model/task_descriptor.py` | `RIN` |
| Transformer backbone components | `TransformerBlock`, `Attention`, `FeedForward` | `model/transformer.py` | Same class names |

`MSMoE` is the Python identifier for the paper's MS-MoE abbreviation. `RCFPM`
is the identifier for RC-FPM's modulation/fusion stage; AllFreqNet invokes
`RegionAdaptiveFrequencyDecoupling` immediately before it. Together these two
stages implement the existing RC-FPM computation. They remain separately
registered to preserve checkpoint keys.

```python
from model import AllFreqNet
from model.ms_moe import MSMoE
from model.rc_fpm import RegionAdaptiveFrequencyDecoupling, RCFPM

model = AllFreqNet()
```

Only class identifiers, file organization and documentation changed. Registered
parameter attributes such as `dict_generator`, `spatial_routing_encoder_level1`,
`frequency_extraction_latent`, `frequency_prompt_module_latent`, `FSPG_high`
and `FSPG_low` retain their existing names. Existing plain state dictionaries
therefore load without key conversion. All training/evaluation/inference entry
points import the real `AllFreqNet` class through the `model` package.

Source code importing the removed `model.Model_AMIR` module must update its
import to `from model import AllFreqNet`. Full pickled model objects tied to old
class paths are outside this compatibility guarantee; use plain state dictionaries.
Historical AMIR mentions in acknowledgments and the previous-class column are
retained for accurate attribution and migration documentation.
