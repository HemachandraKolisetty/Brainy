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

def temporal_encoding(features, num_steps, device='cpu'):
    """
    Temporal encoding where each feature fires a single spike at a time step
    inversely proportional to its value (higher value -> earlier spike).
    """
    features = torch.tensor(features, dtype=torch.float32, device=device)
    
    # Calculate spike times: higher feature value -> earlier spike
    spike_times = torch.round((1.0 - features) * (num_steps - 1)).long()
    spike_times = torch.clamp(spike_times, 0, num_steps - 1) # TODO: remove this line

    batch_size, num_features = spike_times.size()
    spikes = torch.zeros(batch_size, num_steps, num_features, device=device)
    
    # Vectorized assignment of spikes
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_features)
    feature_indices = torch.arange(num_features, device=device).unsqueeze(0).repeat(batch_size, 1)
    spikes[batch_indices, spike_times, feature_indices] = 1.0
    
    return spikes