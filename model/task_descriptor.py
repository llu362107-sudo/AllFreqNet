"""CNN global task descriptor with auxiliary task classification.

See docs/NAMING.md for the paper-to-code mapping and checkpoint conventions.
"""

from torch import nn


class GlobalTaskDescriptor(nn.Module):
    """Extract a global task descriptor and four-way auxiliary class logits."""

    def __init__(self, in_dim, atom_num=16, atom_dim=256):
        super(GlobalTaskDescriptor, self).__init__()
        hidden_dim = 64
        self.CondNet = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True), nn.Conv2d(hidden_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True), nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(hidden_dim, 32, 1))
        self.classifier = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        class_logits = self.classifier(out)
        return (out, class_logits)
