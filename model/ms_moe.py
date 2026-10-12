"""Multi-Scale Mixture-of-Experts (MS-MoE) routing components.

See docs/NAMING.md for the paper-to-code mapping and checkpoint conventions.
"""

import numpy as np
import torch
from torch import nn
from einops import rearrange
from torch.distributions.normal import Normal


class ExpertMLP(nn.Module):

    def __init__(self, in_feat, h_feat=None, out_feat=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_feat = out_feat or in_feat
        h_feat = h_feat or in_feat
        self.fc1 = nn.Linear(in_feat, h_feat)
        self.act = act_layer()
        self.fc2 = nn.Linear(h_feat, out_feat)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SparseExpertDispatcher(object):
    """Dispatch active tokens to experts; retain the original log-domain merge."""

    def __init__(self, num_experts, gates):
        """Create a sparse expert dispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class TaskAdaptiveMoE(nn.Module):
    """Top-2 local expert routing with a shared global expert branch."""

    def __init__(self, input_size, output_size, mlp_ratio, num_experts, noisy_gating=True, use_experts=2):
        super(TaskAdaptiveMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = use_experts
        self.experts = nn.ModuleList([ExpertMLP(input_size, h_feat=int(input_size * mlp_ratio), out_feat=output_size) for i in range(self.num_experts)])
        self.universal_experts = ExpertMLP(input_size, h_feat=int(input_size * mlp_ratio), out_feat=output_size)
        self.w_gate = nn.Parameter(torch.randn(2 * input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(2 * input_size, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer('mean', torch.tensor([0.0]))
        self.register_buffer('std', torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=0.01):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return (gates, load)

    def forward(self, x, prompt):
        B, C, H, W = x.shape
        prompt = prompt.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        x = rearrange(x, 'b c h w -> (b h w) c')
        y_universal = self.universal_experts(x)
        prompt = rearrange(prompt, 'b c h w -> (b h w) c')
        x_gating = torch.cat((x, prompt), dim=1)
        gates, load = self.noisy_top_k_gating(x_gating, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        dispatcher = SparseExpertDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        y = (y + y_universal) / 2.0
        y = rearrange(y, '(b h w) c -> b c h w', b=B, h=H, w=W)
        return (y, loss)


class ChannelRouting(nn.Module):

    def __init__(self, atom_dim, dim):
        super(ChannelRouting, self).__init__()
        self.fc = nn.Linear(atom_dim, dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x, prompt):
        gating_factors = torch.sigmoid(self.fc(prompt))
        gating_factors = gating_factors.unsqueeze(-1).unsqueeze(-1)
        out = x * self.gamma + self.beta
        out = out * gating_factors
        return x + out


class MSMoE(nn.Module):

    def __init__(self, atom_dim, dim, ffn_expansion_factor):
        super(MSMoE, self).__init__()
        self.fc = nn.Linear(atom_dim, dim)
        self.moe = TaskAdaptiveMoE(dim, dim, mlp_ratio=ffn_expansion_factor, num_experts=4, noisy_gating=True, use_experts=2)

    def forward(self, x, prompt):
        d = self.fc(prompt)
        out, loss = self.moe(x, d)
        return (out + x, loss)
