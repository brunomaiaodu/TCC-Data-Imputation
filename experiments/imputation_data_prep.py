import argparse
import logging
from typing import Mapping, Optional, Type, List, Tuple, Union, Any, Dict

import netCDF4 as nc
import numpy as np
import torch
from tsl.ops.imputation import sample_mask
from tsl.engines import Imputer
from tsl.nn.models import SPINModel
from tsl.metrics import torch as torch_metrics

logger = logging.getLogger("imputation")

def prepare_roms_tensor(nc_ds, use_gpu=True,
                        start_time_idx=0, end_time_idx=None, stride_time=1,
                        start_eta=0, end_eta=None, stride_eta=1,
                        start_xi=0, end_xi=None, stride_xi=1):
    """
    Extract main variables from the roms_brasil_se simulation and prepare a torch tensor for processing.
    Args:
        nc_ds: netCDF4 dataset containing target variables;
        use_gpu: whether to try loading the tensor to GPU for faster processing;
        start_time_idx, end_time_id, stride_time: indices for filtering the temporal
            dimension, following Python's array slicing conventions;
        start_eta, end_eta, stride_eta: indices for filtering the eta spatial
            dimension (u axis);
        start_xi, end_xi, stride_xi: likewise for the xi spatial dimension (v axis).
    Returns:
        a torch.Tensor with shape: (time_steps, latitude, longitude, variables)
        where variables (by index):
            0: barotropic northward current velocity, in m/s (north-aligned vbar, interpolated at rho points)
            1: barotropic eastward current velocity,in m/s (east-aligned ubar, interpolated at rho points)
            2: sea surface height above geopotential datum, in m (zeta)
            3: bathymetry, in m (h)
    """

    if (end_time_idx is None):
        end_time_idx = nc_ds['ubar'].shape[0]
    if (end_eta is None):
        end_eta = nc_ds['ubar'].shape[1]
    if (end_xi is None):
        end_xi = nc_ds['ubar'].shape[2]

    device = torch.device('cuda:0') if (torch.cuda.is_available() and use_gpu) else torch.device('cpu')
        
    ubar = torch.tensor(data=nc_ds['ubar'][start_time_idx:end_time_idx:stride_time,
                        :,:]).to(device)
    vbar = torch.tensor(data=nc_ds['vbar'][start_time_idx:end_time_idx:stride_time,
                        :,:]).to(device)
    mask_u = torch.tensor(data=nc_ds['mask_u'][:], dtype=torch.bool).to(device)
    mask_v = torch.tensor(data=nc_ds['mask_v'][:], dtype=torch.bool).to(device)
    mask_rho = torch.tensor(data=nc_ds['mask_rho'][:], dtype=torch.bool).to(device)
    
    left_u_mask = torch.cat([mask_u[:, 0].reshape(-1,1), mask_u], dim=1)
    right_u_mask = torch.cat([mask_u, mask_u[:, -1].reshape(-1,1)], dim=1)
    mask_u_on_rho = (left_u_mask & right_u_mask)
    mask_u_on_rho = (mask_u_on_rho & mask_rho)

    top_v_mask = torch.cat([mask_v[0, :].reshape(1,-1), mask_v], dim=0)
    bottom_v_mask = torch.cat([mask_v, mask_v[-1, :].reshape(1,-1)], dim=0)
    mask_v_on_rho = (top_v_mask & bottom_v_mask)
    mask_v_on_rho = (mask_v_on_rho & mask_rho)
    
    # interpolation from u and v points to rho points. In the extremities, the nearest value is considered.
    # Elsewhere, we take the mean value between the two nearest points.
    left_ubar = torch.cat([ubar[:, :, 0].reshape(ubar.shape[0],-1,1), ubar], dim=2)
    right_ubar = torch.cat([ubar, ubar[:, :, -1].reshape(ubar.shape[0],-1,1)], dim=2)
    ubar_rho = 0.5 * (left_ubar + right_ubar)

    top_vbar = torch.cat([vbar[:, 0, :].reshape(vbar.shape[0],1,-1), vbar], dim=1)
    bottom_vbar = torch.cat([vbar, vbar[:, -1, :].reshape(vbar.shape[0],1,-1)], dim=1)
    vbar_rho = 0.5 * (top_vbar + bottom_vbar)

    ubar_rho = torch.where(mask_u_on_rho, ubar_rho, torch.nan)
    vbar_rho = torch.where(mask_v_on_rho, vbar_rho, torch.nan)
    
    zeta = torch.tensor(data=nc_ds['zeta'][start_time_idx:end_time_idx:stride_time, :, :]).to(device)
    zeta = torch.where(mask_rho, zeta, torch.nan)

    h = torch.tensor(data=nc_ds['h'][start_eta:end_eta:stride_eta,
                     start_xi:end_xi:stride_xi]).to(device)

    # Create data list with all variables
    tensor_data = [
        vbar_rho[:,start_eta:end_eta:stride_eta, start_xi:end_xi:stride_xi],  # 0: vbar
        ubar_rho[:,start_eta:end_eta:stride_eta, start_xi:end_xi:stride_xi],  # 1: ubar
        zeta[:,start_eta:end_eta:stride_eta, start_xi:end_xi:stride_xi],      # 2: sea surface height
        h.expand(vbar_rho.shape[0], -1, -1)     # 3: bathymetry
    ]

    # Convert to torch tensor 
    result = torch.stack(tensor_data, dim=0)

    # Put channels in last dimension
    return torch.movedim(result, 0, -1)


def tsl_dataset(st_tensor):
    """
    Set up a torch spatiotemporal dataset from a tensor.
    """
    pass
    
     

def gnn_style_tensor(st_tensor: torch.Tensor):
    """
    Modify a spatiotemporal grid-like dataset for application in graph neural network models.

    Arguments:
        st_tensor: torch tensor with shape (time, x-axis, y-axis, channels)
    Returns:
        gnn_t: torch tensor with shape (batch, time, channels)
        A: torch tensor representing the adjacency matrix of the spatial grid.

    """
    T, Sx, Sy, C = st_tensor.shape
    A = torch.zeros((Sx * Sy, Sx * Sy), dtype=torch.int32)
    
    # Create adjacency matrix for 8-connected grid
    for i in range(Sx):
        for j in range(Sy):
            # Get current node index
            node_idx = i * Sy + j
            
            # Check all 8 possible directions
            directions = [
                (i-1, j),      # up
                (i+1, j),      # down
                (i, j-1),     # left
                (i, j+1),     # right
                (i-1, j-1),   # up-left
                (i-1, j+1),   # up-right
                (i+1, j-1),   # down-left
                (i+1, j+1)    # down-right
            ]
            
            # Check each direction
            for ni, nj in directions:
                if 0 <= ni < Sx and 0 <= nj < Sy:
                    neighbor_idx = ni * Sy + nj
                    A[node_idx, neighbor_idx] = 1
                    A[neighbor_idx, node_idx] = 1  # Make symmetric

    gnn_t = st_tensor.reshape((T, Sx * Sy, C))
    return gnn_t, A.to(gnn_t.device)

def generate_downscaling_mask(shape, stride=3):
    """
    Generate a downscaling mask for 2D spatial data.
    
    Args:
        shape: tuple (batch_size, time_steps, height, width, variables)
        stride: int, how far apart sampled (observed) pixels are
        
    Returns:
        mask: same shape, with 1s for observed and 0s for masked
    """
    B, T, H, W, C = shape
    mask = np.zeros((B, T, H, W, C), dtype=np.uint8)

    for h in range(0, H, stride):
        for w in range(0, W, stride):                                                                                                                                                   
            mask[:, :, h, w, :] = 1  # observe every stride-th pixel

    return mask


def missing_data_mask(
        shape: torch.Size,
        p:float,
        min_sequence: int = 1,
        max_sequence: int = 1,
        rng: np.random.Generator | None = None) -> torch.Tensor:

    if rng is None:
        rng = np.random.default_rng(0x7E57)

    return sample_mask(
        shape = shape,
        p = 2 * p / (max_sequence - min_sequence),
        p_noise = 0,
        min_seq = min_sequence,
        max_seq = max_sequence,
        rng = rng
        )


def prepare_stts_train_test(data, sequence_length=30, overlap=15, 
                           test_split_dim="time", test_split_method="cutoff",
                           test_split_value=0.8, random_seed=42):
    """
    Prepare space-time series data for training and testing with sliding windows.
    
    Args:
        data: torch.Tensor containing the space-time series dataset
        sequence_length: int, length of the sliding window sequence
        overlap: int, overlap between consecutive windows
        test_split_dim: str, dimension to split test set ("time", "latitude", "longitude")
        test_split_method: str, method for test set split ("cutoff", "chunk", "random")
        test_split_value: float or int, value for the split method (0-1 for random, cutoff point for others)
        random_seed: int, seed for random operations
        
    Returns:
        tuple: (train_dataset, test_dataset) where each is a PyTorch Dataset
    """
    
    # Create sliding windows
    def create_sliding_windows(data, seq_len, overlap):
        X = []
        for i in range(0, len(data), seq_len - overlap):
            window = data[i:i+seq_len]
            if len(window) == seq_len:
                X.append(window[:])
        return torch.stack(X, dim=0)
    
    # Create sliding windows
    X = create_sliding_windows(data, sequence_length, overlap)
    
    # Split into train/test based on the specified dimension
    if test_split_dim == "time":
        dim = 0
    elif test_split_dim == "latitude":
        dim = 1
    elif test_split_dim == "longitude":
        dim = 2
    else:
        raise ValueError(f"Invalid dimension for test split: {test_split_dim}")

    if test_split_method == "random":
        np.random.seed(random_seed)
        train_idx = np.random.choice(X.shape[dim], int(X.shape[dim]*test_split_value), replace=False)
        test_idx = np.setdiff1d(np.arange(X.shape[dim]), train_idx)
    else:
        # Cutoff or chunk-based split
        test_start = int(X.shape[dim] * test_split_value)
        train_idx = np.arange(test_start)
        test_idx = np.arange(test_start, X.shape[dim])
        
    if dim == 0:
        X_train, X_test = X[train_idx, :, :], X[test_idx, :, :]
    elif dim == 1:
        X_train, X_test = X[:, train_idx, :], X[:, test_idx, :]
    else:  # dim == 2
        X_train, X_test = X[:, :, train_idx], X[:, :, test_idx]
        
    # Create PyTorch datasets
    class SpaceTimeDataset(torch.utils.data.Dataset):
        def __init__(self, X):
            self.X = X
            
        def __len__(self):
            return len(self.X)
            
        def __getitem__(self, idx):
            return self.X[idx]
            
    train_dataset = SpaceTimeDataset(X_train)
    test_dataset = SpaceTimeDataset(X_test)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader