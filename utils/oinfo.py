import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma, gamma
from tqdm.auto import tqdm
import torch.distributed as dist
from typing import Tuple, Optional

def entropy_knn(X, k=3):
    """
    Estimates differential entropy H(X) using the Kozachenko-Leonenko estimator.
    X: (N, d) array of samples.
    k: number of nearest neighbors.
    """
    N, d = X.shape
    
    # Small noise to avoid zero distances if there are duplicates
    X = X + 1e-10 * np.random.randn(N, d)
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='chebyshev').fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # Distance to the k-th nearest neighbor
    # distances[:, k-1] is the distance to the k-th neighbor (0-indexed, so k-1)
    # But kneighbors returns k neighbors including the point itself if it's in the set?
    # sklearn kneighbors returns the point itself as the first neighbor (dist=0) if it's in the data.
    # So we need k+1 neighbors and take the last one.
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='chebyshev').fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # The distance to the k-th neighbor is at index k (0 is self)
    rho = distances[:, k]
    
    # Volume of unit ball in d dimensions for Chebyshev metric (L_infinity) is (2)^d?
    # Wait, Kozachenko-Leonenko usually assumes Euclidean (L2) or Maximum (L_inf).
    # For L_inf (Chebyshev), the volume of a ball of radius r is (2r)^d.
    # c_d = 2^d? No, log(c_d) term.
    # Let's stick to Euclidean (L2) which is standard.
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean').fit(X)
    distances, _ = nbrs.kneighbors(X)
    rho = distances[:, k]
    
    # Volume of unit ball in d dimensions (Euclidean)
    # V_d = pi^(d/2) / gamma(d/2 + 1)
    # log(V_d) = (d/2) * log(pi) - log(gamma(d/2 + 1))
    
    log_cd = (d/2) * np.log(np.pi) - np.log(gamma(d/2 + 1))
    
    # Entropy estimate
    # H(X) = psi(N) - psi(k) + log(c_d) + (d/N) * sum(log(rho))
    
    # Avoid log(0)
    rho = np.maximum(rho, 1e-15)
    
    H = digamma(N) - digamma(k) + log_cd + (d/N) * np.sum(np.log(rho))
    return H

def calculate_oinfo(X, k=3):
    """
    Calculates O-information for the variables in X (N, d).
    O(X) = (d-2)H(X) + sum_{i=1}^d [H(X_i) - H(X_{-i})]
    """
    N, d = X.shape
    
    # H(X)
    H_X = entropy_knn(X, k)
    
    sum_term = 0
    for i in range(d):
        # H(X_i)
        X_i = X[:, [i]]
        H_Xi = entropy_knn(X_i, k)
        
        # H(X_{-i})
        # Create mask for all columns except i
        mask = np.ones(d, dtype=bool)
        mask[i] = False
        X_minus_i = X[:, mask]
        H_X_minus_i = entropy_knn(X_minus_i, k)
        
        sum_term += (H_Xi - H_X_minus_i)
        
    oinfo = (d - 2) * H_X + sum_term
    return oinfo

def calculate_oinfo_gradient(X, k=3):
    """
    Calculates the 'local gradient' of O-information for each neuron.
    Defined as Delta Omega_j = Omega(X) - Omega(X_{-j}).
    Positive value means the neuron contributes to redundancy.
    
    Supports distributed computation across multiple GPUs (by neurons, not by data).
    """
    if not isinstance(X, torch.Tensor):
        X_t = torch.tensor(X, dtype=torch.float32)
    else:
        X_t = X.float()

    device = X_t.device if X_t.is_cuda else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    X_t = X_t.to(device)

    def torch_entropy_knn(X_t, k=3, eps=1e-12):
        """
        Estimador KL en torch (L2). Complejidad O(N^2); asumimos N moderado (~1000).
        """
        N, d = X_t.shape
        X_noise = X_t + eps * torch.randn_like(X_t)
        dist = torch.cdist(X_noise, X_noise, p=2)
        knn_dist, _ = torch.topk(dist, k + 1, dim=1, largest=False)
        rho = knn_dist[:, k] + eps

        log_cd = (d / 2) * torch.log(torch.tensor(np.pi, device=device)) - torch.lgamma(torch.tensor(d / 2 + 1.0, device=device))
        H = (
            torch.digamma(torch.tensor(float(N), device=device))
            - torch.digamma(torch.tensor(float(k), device=device))
            + log_cd
            + (d / N) * torch.sum(torch.log(rho))
        )
        return H

    def torch_calculate_oinfo(X_t, k=3):
        N, d = X_t.shape
        H_X = torch_entropy_knn(X_t, k)
        sum_term = 0.0
        for i in range(d):
            H_Xi = torch_entropy_knn(X_t[:, i : i + 1], k)
            mask = torch.ones(d, dtype=torch.bool, device=device)
            mask[i] = False
            H_X_minus_i = torch_entropy_knn(X_t[:, mask], k)
            sum_term = sum_term + (H_Xi - H_X_minus_i)
        oinfo = (d - 2) * H_X + sum_term
        return oinfo

    omega_X = torch_calculate_oinfo(X_t, k)
    grads = []
    d = X_t.shape[1]
    
    desc = "OI grad por capa"
    for j in tqdm(range(d), desc=desc, leave=False):
        mask = torch.ones(d, dtype=torch.bool, device=device)
        mask[j] = False
        omega_minus_j = torch_calculate_oinfo(X_t[:, mask], k)
        grads.append((omega_X - omega_minus_j).item())

    grads_np = np.array(grads, dtype=np.float64)
    return grads_np, omega_X.item()


def calculate_oinfo_gradient_distributed(X, k=3) -> Tuple[np.ndarray, float]:
    """
    Distributed OInfo calculation across multiple GPUs.
    Distributes neuron dimension across GPUs, not the batch dimension.
    
    Each GPU computes OInfo gradient for a subset of neurons independently.
    Activations X are shared across all GPUs (broadcast).
    
    Parameters:
    -----------
    X : np.ndarray or torch.Tensor
        Shape (N, d) where N=samples, d=hidden_dim (e.g., 8960 for Qwen)
    k : int
        Number of neighbors for KNN entropy estimation
    
    Returns:
    --------
    grads : np.ndarray
        Shape (d,) - OInfo gradient for each neuron
    omega_X : float
        Global OInfo value
    
    Example:
    --------
    # Qwen with 8960 hidden dimensions across 4 GPUs
    activations.shape = (1000, 8960)
    grads, omega = calculate_oinfo_gradient_distributed(activations)
    # GPU0: computes neurons 0-2239
    # GPU1: computes neurons 2240-4479
    # GPU2: computes neurons 4480-6719
    # GPU3: computes neurons 6720-8959
    # Results gathered back to GPU0
    """
    
    is_distributed = dist.is_available() and dist.is_initialized()
    
    if not is_distributed:
        return calculate_oinfo_gradient(X, k)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    
    if not isinstance(X, torch.Tensor):
        X_t = torch.tensor(X, dtype=torch.float32)
    else:
        X_t = X.float()
    
    X_t = X_t.to(device)
    N, d = X_t.shape
    
    # Synchronize before starting computation to ensure all ranks are ready
    dist.barrier()
    
    def torch_entropy_knn(X_t, k=3, eps=1e-12):
        N, d = X_t.shape
        X_noise = X_t + eps * torch.randn_like(X_t)
        dist_mat = torch.cdist(X_noise, X_noise, p=2)
        knn_dist, _ = torch.topk(dist_mat, k + 1, dim=1, largest=False)
        rho = knn_dist[:, k] + eps

        log_cd = (d / 2) * torch.log(torch.tensor(np.pi, device=device)) - torch.lgamma(torch.tensor(d / 2 + 1.0, device=device))
        H = (
            torch.digamma(torch.tensor(float(N), device=device))
            - torch.digamma(torch.tensor(float(k), device=device))
            + log_cd
            + (d / N) * torch.sum(torch.log(rho))
        )
        return H

    def torch_calculate_oinfo(X_t, k=3):
        N, d = X_t.shape
        H_X = torch_entropy_knn(X_t, k)
        sum_term = 0.0
        for i in range(d):
            H_Xi = torch_entropy_knn(X_t[:, i : i + 1], k)
            mask = torch.ones(d, dtype=torch.bool, device=device)
            mask[i] = False
            H_X_minus_i = torch_entropy_knn(X_t[:, mask], k)
            sum_term = sum_term + (H_Xi - H_X_minus_i)
        oinfo = (d - 2) * H_X + sum_term
        return oinfo

    omega_X = torch_calculate_oinfo(X_t, k)
    
    neurons_per_gpu = (d + world_size - 1) // world_size
    start_idx = rank * neurons_per_gpu
    end_idx = min((rank + 1) * neurons_per_gpu, d)
    local_neurons = range(start_idx, end_idx)
    
    local_grads = []
    desc = f"OI grad GPU{rank} [{start_idx}-{end_idx}]"
    for j in tqdm(local_neurons, desc=desc, leave=False):
        mask = torch.ones(d, dtype=torch.bool, device=device)
        mask[j] = False
        omega_minus_j = torch_calculate_oinfo(X_t[:, mask], k)
        local_grads.append((omega_X - omega_minus_j).item())
    
    # Convert local gradients to tensor for all_gather
    # Pad to same length across all GPUs
    local_tensor = torch.zeros(neurons_per_gpu, device=device, dtype=torch.float64)
    local_tensor[:len(local_grads)] = torch.tensor(local_grads, device=device, dtype=torch.float64)
    
    # Synchronize before collective operation
    dist.barrier()
    
    # Use all_gather with tensors (more reliable than gather_object)
    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, local_tensor)
    
    # Combine results
    grads_list = []
    for i, tensor in enumerate(gathered_tensors):
        # Calculate how many valid gradients this rank contributed
        gpu_start = i * neurons_per_gpu
        gpu_end = min((i + 1) * neurons_per_gpu, d)
        num_valid = gpu_end - gpu_start
        grads_list.extend(tensor[:num_valid].cpu().numpy().tolist())
    
    grads_np = np.array(grads_list[:d], dtype=np.float64)
    
    # Final synchronization
    dist.barrier()
    
    return grads_np, omega_X.item()


def calculate_oinfo_gradient_auto(X, k=3) -> Tuple[np.ndarray, float]:
    """
    Automatically selects between standard and distributed OInfo calculation.
    
    Uses distributed computation if:
    - Multiple GPUs are available and initialized
    - Neuron dimension is large (> 1000)
    
    Otherwise falls back to standard single-GPU computation.
    
    Parameters:
    -----------
    X : np.ndarray or torch.Tensor
        Shape (N, d) where N=samples, d=hidden_dim
    k : int
        Number of neighbors for KNN entropy estimation
    
    Returns:
    --------
    grads : np.ndarray
        Shape (d,) - OInfo gradient for each neuron
    omega_X : float
        Global OInfo value
    """
    is_distributed = dist.is_available() and dist.is_initialized()
    num_neurons = X.shape[1] if isinstance(X, np.ndarray) else X.shape[1]
    
    if is_distributed and num_neurons > 1000:
        return calculate_oinfo_gradient_distributed(X, k)
    else:
        return calculate_oinfo_gradient(X, k)
