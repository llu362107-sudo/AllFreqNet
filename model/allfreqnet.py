"""AllFreqNet restoration network. Parameter attribute names remain checkpoint-compatible.

See docs/NAMING.md for the paper-to-code mapping and checkpoint conventions.
"""

import torch
from torch import nn

from .ms_moe import MSMoE
from .rc_fpm import RegionAdaptiveFrequencyDecoupling, RCFPM
from .task_descriptor import GlobalTaskDescriptor
from .transformer import Downsample, OverlapPatchEmbed, TransformerBlock, Upsample


class AllFreqNet(nn.Module):
    """AllFreqNet with paper-aligned class names and historical parameter keys."""

    def __init__(self, inp_channels=1, out_channels=1, dim=42, num_blocks=[5, 7, 7, 9], num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', dual_pixel_task=False):
        super(AllFreqNet, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        atom_dim = 256
        atom_num = 32
        self.dict_generator = GlobalTaskDescriptor(in_dim=inp_channels, atom_num=atom_num, atom_dim=atom_dim)
        self.spatial_routing_encoder_level1 = MSMoE(atom_dim=32, dim=dim, ffn_expansion_factor=ffn_expansion_factor)
        self.spatial_routing_encoder_level2 = MSMoE(atom_dim=32, dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor)
        self.spatial_routing_encoder_level3 = MSMoE(atom_dim=32, dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.frequency_extraction_latent = RegionAdaptiveFrequencyDecoupling(int(dim * 2 ** 3), kernel_size=3, stride=1, group=8)
        self.frequency_prompt_module_latent = RCFPM(dim=int(dim * 2 ** 3), k_size=3)
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.frequency_extraction_3 = RegionAdaptiveFrequencyDecoupling(int(dim * 2 ** 2), kernel_size=3, stride=1, group=8)
        self.frequency_prompt_module_3 = RCFPM(dim=int(dim * 2 ** 2), k_size=3)
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.frequency_extraction_2 = RegionAdaptiveFrequencyDecoupling(int(dim * 2 ** 1), kernel_size=3, stride=1, group=12)
        self.frequency_prompt_module_2 = RCFPM(dim=int(dim * 2 ** 1), k_size=3)
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.frequency_extraction_1 = RegionAdaptiveFrequencyDecoupling(int(dim * 2 ** 1), kernel_size=3, stride=1, group=12)
        self.frequency_prompt_module_1 = RCFPM(dim=int(dim * 2 ** 1), k_size=3)
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        prompt, class_logits = self.dict_generator(inp_img)
        inp_enc_level1 = self.patch_embed(inp_img)
        # Preserve version3: this routed feature is unused; its loss is accumulated.
        # See docs/IMPLEMENTATION_NOTES.md before changing the encoder input.
        inp_enc_level1_moe, loss_tmp = self.spatial_routing_encoder_level1(inp_enc_level1, prompt)
        loss_importance = loss_tmp
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_enc_level2, loss_tmp = self.spatial_routing_encoder_level2(inp_enc_level2, prompt)
        loss_importance = loss_importance + loss_tmp
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        inp_enc_level3, loss_tmp = self.spatial_routing_encoder_level3(inp_enc_level3, prompt)
        loss_importance = loss_importance + loss_tmp
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        latent_low_part_latent, latent_out_high_latent = self.frequency_extraction_latent(latent)
        frequency_prompt_latent = self.frequency_prompt_module_latent(latent_low_part_latent, latent_out_high_latent, latent)
        inp_dec_level3 = self.up4_3(frequency_prompt_latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        latent_low_part_3, latent_out_high_3 = self.frequency_extraction_3(out_dec_level3)
        frequency_propt_3 = self.frequency_prompt_module_3(latent_low_part_3, latent_out_high_3, out_dec_level3)
        inp_dec_level2 = self.up3_2(frequency_propt_3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        latent_low_part_2, latent_out_high_2 = self.frequency_extraction_2(out_dec_level2)
        frequency_propt_2 = self.frequency_prompt_module_2(latent_low_part_2, latent_out_high_2, out_dec_level2)
        inp_dec_level1 = self.up2_1(frequency_propt_2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        latent_low_part_1, latent_out_high_1 = self.frequency_extraction_1(out_dec_level1)
        frequency_propt_1 = self.frequency_prompt_module_1(latent_low_part_1, latent_out_high_1, out_dec_level1)
        out_dec_level1 = self.refinement(frequency_propt_1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        if self.training:
            return (out_dec_level1, loss_importance, class_logits)
        else:
            return out_dec_level1
