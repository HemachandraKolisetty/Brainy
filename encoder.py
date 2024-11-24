# Various encoders for Spiking Neural Networks
import torch

def poisson_encoding(features, num_steps, device='cpu'):
    features_repeated = features.unsqueeze(1).repeat(1, num_steps, 1)
    rand_vals = torch.rand_like(features_repeated, device=device)
    spikes = (rand_vals < features_repeated).float()
    return spikes

def rate_encoding(features, num_steps, device='cpu'):
    features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    spike_trains = poisson_encoding(features_tensor, num_steps=num_steps, device=device)
    return spike_trains