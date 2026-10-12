"""Region-Conditioned Frequency Prompt Modulation (RC-FPM) components.

See docs/NAMING.md for the paper-to-code mapping and checkpoint conventions.
"""

import torch
from torch import nn
import torch.nn.functional as F


def repeat_and_crop(x: torch.Tensor, target_size: tuple) -> torch.Tensor:
    assert len(x.shape) == len(target_size), '目标尺寸维度需与输入张量一致'
    repeat_factors = []
    for orig_dim, target_dim in zip(x.shape, target_size):
        if target_dim <= 0:
            raise ValueError('目标尺寸需为正整数')
        repeats = (target_dim + orig_dim - 1) // orig_dim
        repeat_factors.append(repeats)
    expanded = x.repeat(*repeat_factors)
    slices = [slice(0, dim) for dim in target_size]
    return expanded[slices]


class RegionAdaptiveFrequencyDecoupling(nn.Module):
    """Generate normalized grouped low-pass kernels separately at each position."""

    def __init__(self, inchannels, kernel_size=3, stride=1, group=8, kernel_gen_kernel_size=3):
        super(RegionAdaptiveFrequencyDecoupling, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        pad = kernel_gen_kernel_size // 2
        self.kernel_gen = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=kernel_gen_kernel_size, padding=pad, bias=False)
        self.kernel_gate = nn.Conv2d(group * kernel_size ** 2, group * kernel_size ** 2, kernel_size=1, bias=False)
        self.act_gate = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Softmax(dim=2)
        nn.init.kaiming_normal_(self.kernel_gen.weight, mode='fan_out', nonlinearity='relu')
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

    def forward(self, x):
        n, c, h, w = x.shape
        if c % self.group != 0:
            raise ValueError('inchannels must be divisible by group')
        if self.stride != 1:
            raise ValueError('This version assumes stride=1. Ask if you need stride>1.')
        low_filter = self.kernel_gen(x)
        low_filter = low_filter * self.act_gate(self.kernel_gate(low_filter))
        low_filter = self.bn(low_filter)
        low_filter = low_filter.view(n, self.group, self.kernel_size ** 2, h * w)
        low_filter = self.act(low_filter)
        x_pad = self.pad(x)
        patches = F.unfold(x_pad, kernel_size=self.kernel_size, stride=self.stride)
        patches = patches.view(n, self.group, c // self.group, self.kernel_size ** 2, -1)
        low_part = (patches * low_filter.unsqueeze(2)).sum(dim=3)
        low_part = low_part.view(n, c, h, w)
        out_high = x - low_part
        return (low_part, out_high)


def make_coord_grid(n, h, w, device, dtype):
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(n, 1, h, w)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(n, 1, h, w)
    return torch.cat([xs, ys], dim=1)


class CoordinateConditionedFrequencyPromptGenerator(nn.Module):
    """Modulate a frequency band using content and normalized image coordinates."""

    def __init__(self, dim=3, flag_highF=True, k_size=3, hidden_mult=1):
        super().__init__()
        self.flag_highF = flag_highF
        self.body = nn.Sequential(nn.Conv2d(dim, dim, (1, k_size), padding=(0, k_size // 2), groups=dim, bias=False), nn.Conv2d(dim, dim, (k_size, 1), padding=(k_size // 2, 0), groups=dim, bias=False), nn.GELU())
        hidden = max(dim * hidden_mult, dim)
        self.weight_gen = nn.Sequential(nn.Conv2d(dim + 2, hidden, kernel_size=1, bias=True), nn.GELU(), nn.Conv2d(hidden, dim, kernel_size=1, bias=True), nn.Sigmoid())

    def forward(self, ffm, H, W, TEST=False):
        mode = 'bilinear' if self.flag_highF else 'bicubic'
        ffm = F.interpolate(ffm, size=(H, W), mode=mode, align_corners=False)
        y_att = self.body(ffm)
        y_f = y_att * ffm
        n, c, h, w = y_f.shape
        coord = make_coord_grid(n, h, w, device=y_f.device, dtype=y_f.dtype)
        weight = self.weight_gen(torch.cat([y_f, coord], dim=1))
        y = y_f * weight
        return y


class RCFPM(nn.Module):
    """Coordinate-conditioned modulation and fusion stage of RC-FPM.

    AllFreqNet calls RegionAdaptiveFrequencyDecoupling separately before this
    stage, retaining the original checkpoint attribute hierarchy.
    """

    def __init__(self, dim, k_size=3):
        super().__init__()
        self.FSPG_high = CoordinateConditionedFrequencyPromptGenerator(dim=dim, flag_highF=True, k_size=k_size)
        self.FSPG_low = CoordinateConditionedFrequencyPromptGenerator(dim=dim, flag_highF=False, k_size=k_size)
        self.simple_Fusion = nn.Conv2d(3 * dim, dim, kernel_size=1, stride=1)

    def forward(self, low_part, out_high, x):
        b, c, h, w = x.shape
        y_h = self.FSPG_high(out_high, h, w)
        y_l = self.FSPG_low(low_part, h, w)
        frequency_prompt = torch.cat([y_h, y_l, x], dim=1)
        frequency_prompt = self.simple_Fusion(frequency_prompt)
        return frequency_prompt
