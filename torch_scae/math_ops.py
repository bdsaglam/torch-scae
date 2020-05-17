import torch


def log_safe(tensor, eps=1e-16):
    is_zero = tensor < eps
    tensor = torch.where(is_zero, torch.ones_like(tensor), tensor)
    tensor = torch.where(is_zero, torch.zeros_like(tensor) - 1e8, torch.log(tensor))
    return tensor


def cross_entropy_safe(labels, probs, dim=-1):
    return torch.mean(-torch.sum(labels * log_safe(probs), dim=dim))


def normalize(tensor, dim):
    return tensor / (torch.sum(tensor, dim, keepdim=True) + 1e-8)
