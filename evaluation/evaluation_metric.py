"""Version3 PSNR/SSIM/RMSE, with rectangular-image and constant-target handling."""

from math import exp

import torch
import torch.nn.functional as F


def compute_MSE(image, target):
    return ((image - target) ** 2).mean()


def compute_RMSE(image, target):
    return torch.sqrt(compute_MSE(image, target)).item()


def compute_PSNR(image, target, data_range):
    if float(data_range) <= 0:
        return float("nan")
    return (10 * torch.log10(torch.as_tensor(data_range, dtype=image.dtype, device=image.device) ** 2 / compute_MSE(image, target))).item()


def gaussian(window_size, sigma):
    values = torch.tensor([exp(-(i - window_size // 2) ** 2 / (2 * sigma ** 2)) for i in range(window_size)])
    return values / values.sum()


def create_window(window_size, channel):
    one_d = gaussian(window_size, 1.5).unsqueeze(1)
    return one_d.mm(one_d.t()).unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size).contiguous()


def compute_SSIM(image, target, data_range, window_size=11):
    if float(data_range) <= 0:
        return float("nan")
    if image.ndim == 2:
        image, target = image[None, None], target[None, None]
    if image.ndim == 3:
        image, target = image[None], target[None]
    channels = image.shape[1]
    window = create_window(window_size, channels).to(image)
    def convolve(value):
        return F.conv2d(value, window, padding=window_size // 2, groups=channels)
    mu1, mu2 = convolve(image), convolve(target)
    variance1 = convolve(image * image) - mu1 ** 2
    variance2 = convolve(target * target) - mu2 ** 2
    covariance = convolve(image * target) - mu1 * mu2
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mu1 * mu2 + c1) * (2 * covariance + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (variance1 + variance2 + c2))
    return score.mean().item()


def compute_measure(image, target, data_range):
    return compute_PSNR(image, target, data_range), compute_SSIM(image, target, data_range), compute_RMSE(image, target)
