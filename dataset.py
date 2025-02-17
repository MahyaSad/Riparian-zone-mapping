# dataset.py
import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import cupy as cp
import zarr
from tqdm import tqdm
import random

class EnhancedZarrTileDataset(Dataset):
    def __init__(self, base_path, tile_ids, patch_size=64, timestamps_file=None, 
                 label_variable=None, no_data_value=-9999, pad_mode='both', 
                 stats_file=None, calculate_stats=False, gpu_id=0,
                 balanced_sampling=True, samples_per_tile=100):
        super().__init__()
        self.base_path = base_path
        self.tile_ids = tile_ids
        self.patch_size = patch_size
        self.label_variable = label_variable
        self.no_data_value = no_data_value
        self.pad_mode = pad_mode
        self.gpu_id = gpu_id
        self.balanced_sampling = balanced_sampling
        self.samples_per_tile = samples_per_tile
        
        # Define variables
        self.s1_variables = ['VH', 'VV', 'incidence_angle', 'DEM']
        self.s2_variables = ['B02', 'B03', 'B04', 'B08', 'B11', 
                           'NDVI', 'NDWI_veg', 'NDWI_water', 'DEM']
        
        # Initialize statistics
        if stats_file and os.path.exists(stats_file):
            self.stats = self._load_statistics(stats_file)
        elif calculate_stats:
            print("Calculating normalization statistics...")
            self.stats = self._calculate_statistics_gpu()
            if stats_file:
                self._save_statistics(stats_file)
        
        # Setup CUDA
        self.stream = cp.cuda.Stream(non_blocking=True)
        
        # Load timestamps
        self.timestamps = {}
        if timestamps_file and os.path.exists(timestamps_file):
            self.timestamps = cp.load(timestamps_file)
        
        # Initialize dataset
        self.max_T = self._calculate_max_T()
        self._precompute_patch_locations()

    def _calculate_max_T(self):
        max_T = 0
        for tile_id in self.tile_ids:
            sample_path = os.path.join(self.base_path, f"{tile_id}_VH.zarr")
            if os.path.exists(sample_path):
                with zarr.open(sample_path, mode='r') as z:
                    T = z.shape[0]
                    if T > max_T:
                        max_T = T
        return max_T

    def _precompute_patch_locations(self):
        self.patch_locations = {'class_0': [], 'class_1': []}
        
        for tile_id in self.tile_ids:
            label_path = os.path.join(self.base_path, f"{tile_id}_{self.label_variable}.zarr")
            if not os.path.exists(label_path):
                continue
                
            with zarr.open(label_path, mode='r') as z:
                labels = cp.array(z[:])
                H, W = labels.shape
                
                # Sliding window with 50% overlap
                for y in range(0, H - self.patch_size + 1, self.patch_size // 2):
                    for x in range(0, W - self.patch_size + 1, self.patch_size // 2):
                        patch = labels[y:y+self.patch_size, x:x+self.patch_size]
                        
                        # Validate patch
                        valid_pixels = (patch != self.no_data_value).sum()
                        if valid_pixels < 0.8 * self.patch_size * self.patch_size:
                            continue
                        
                        # Count class pixels
                        class_0_pixels = (patch == 0).sum()
                        class_1_pixels = (patch == 1).sum()
                        
                        min_pixels = 0.2 * self.patch_size * self.patch_size
                        if class_0_pixels > min_pixels or class_1_pixels > min_pixels:
                            location = {'tile_id': tile_id, 'y': y, 'x': x}
                            if class_0_pixels > class_1_pixels:
                                self.patch_locations['class_0'].append(location)
                            else:
                                self.patch_locations['class_1'].append(location)
        
        # Calculate sampling weights
        self.class_weights = {
            'class_0': 1.0 / max(len(self.patch_locations['class_0']), 1),
            'class_1': 1.0 / max(len(self.patch_locations['class_1']), 1)
        }
        
        total_weight = sum(self.class_weights.values())
        self.class_weights = {k: v/total_weight for k, v in self.class_weights.items()}
        
        total_samples = len(self.patch_locations['class_0']) + len(self.patch_locations['class_1'])
        self.samples_per_epoch = min(total_samples, self.samples_per_tile * len(self.tile_ids))

    def _calculate_statistics_gpu(self):
        stats = {}
        with cp.cuda.Device(self.gpu_id):
            for var_name in self.s1_variables + self.s2_variables:
                if var_name not in stats:
                    is_index = var_name in ['NDVI', 'NDWI_veg', 'NDWI_water']
                    if is_index:
                        stats[var_name] = {'mean': 0.0, 'std': 1.0}
                    else:
                        values_list = []
                        for tile_id in tqdm(self.tile_ids, desc=f"Calculating stats for {var_name}"):
                            zarr_path = os.path.join(self.base_path, f"{tile_id}_{var_name}.zarr")
                            if os.path.exists(zarr_path):
                                with zarr.open(zarr_path, mode='r') as z:
                                    data = cp.array(z[:])
                                    valid_mask = data != self.no_data_value
                                    values_list.append(data[valid_mask])
                        with self.stream:
                            all_values = cp.concatenate(values_list)
                            stats[var_name] = {
                                'mean': float(cp.mean(all_values)),
                                'std': float(cp.std(all_values))
                            }
        return stats

    def _normalize_data_gpu(self, data, variable):
        with cp.cuda.Device(self.gpu_id):
            stats = self.stats[variable]
            data_gpu = cp.array(data)
            valid_mask = data_gpu != self.no_data_value
            with self.stream:
                if variable in ['NDVI', 'NDWI_veg', 'NDWI_water']:
                    data_gpu[~valid_mask] = 0

                else:
                    normalized = cp.zeros_like(data_gpu)
                    normalized[valid_mask] = (data_gpu[valid_mask] - stats['mean']) / stats['std']
                    data_gpu = normalized
            return cp.asnumpy(data_gpu)

    def _extract_patch(self, data, y, x):
        """Extract a patch from the data at position (y, x)"""
        if len(data.shape) == 3:  # [T, H, W]
            return data[:, y:y+self.patch_size, x:x+self.patch_size]
        elif len(data.shape) == 2:  # [H, W]
            return data[y:y+self.patch_size, x:x+self.patch_size]
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

    def _pad_time_series(self, data_np):
        """Pad time series to max_T"""
        T, H, W = data_np.shape
        if T < self.max_T:
            pad_width = ((0, self.max_T - T), (0, 0), (0, 0))
            data_np = np.pad(data_np, pad_width, mode='constant', constant_values=self.no_data_value)
        return data_np

    def _load_statistics(self, stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)

    def _save_statistics(self, stats_file):
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Randomly select a class based on weights
        selected_class = np.random.choice(['class_0', 'class_1'], 
                                        p=[self.class_weights['class_0'], 
                                           self.class_weights['class_1']])
        
        # Randomly select a patch location for the chosen class
        patch_info = random.choice(self.patch_locations[selected_class])
        tile_id = patch_info['tile_id']
        y, x = patch_info['y'], patch_info['x']
        
        with cp.cuda.Device(self.gpu_id):
            # Process Sentinel-1 data
            s1_data_list = []
            for var_name in self.s1_variables:
                zarr_path = os.path.join(self.base_path, f"{tile_id}_{var_name}.zarr")
                with zarr.open(zarr_path, mode='r') as z:
                    data = cp.array(z[:])
                    if var_name == 'DEM':
                        if len(data.shape) == 2:
                            patch = self._extract_patch(data, y, x)
                            patch = cp.tile(patch[cp.newaxis, :, :], (self.max_T, 1, 1))
                        else:
                            patch = self._extract_patch(data, y, x)
                    else:
                        data = cp.array(z[:])
                        patch = self._extract_patch(data, y, x)
                        patch = self._pad_time_series(cp.asnumpy(patch))
                    normalized_patch = self._normalize_data_gpu(cp.asnumpy(patch), var_name)
                    s1_data_list.append(normalized_patch)
            
            # Process Sentinel-2 data
            s2_data_list = []
            for var_name in self.s2_variables:
                zarr_path = os.path.join(self.base_path, f"{tile_id}_{var_name}.zarr")
                with zarr.open(zarr_path, mode='r') as z:
                    data = cp.array(z[:])
                    if var_name == 'DEM':
                        if len(data.shape) == 2:
                            patch = self._extract_patch(data, y, x)
                            patch = cp.tile(patch[cp.newaxis, :, :], (self.max_T, 1, 1))
                        else:
                            patch = self._extract_patch(data, y, x)
                    else:
                        data = cp.array(z[:])
                        patch = self._extract_patch(data, y, x)
                        patch = self._pad_time_series(cp.asnumpy(patch))
                    normalized_patch = self._normalize_data_gpu(cp.asnumpy(patch), var_name)
                    s2_data_list.append(normalized_patch)
            
            # Process labels
            label_path = os.path.join(self.base_path, f"{tile_id}_{self.label_variable}.zarr")
            with zarr.open(label_path, mode='r') as z:
                label_data = cp.array(z[:])
                label_patch = self._extract_patch(label_data, y, x)
                label_tensor = torch.from_numpy(cp.asnumpy(label_patch).astype(np.int64))
        
        return (
            torch.from_numpy(np.stack(s1_data_list, axis=0).astype(np.float32)),
            torch.from_numpy(np.stack(s2_data_list, axis=0).astype(np.float32)),
            label_tensor
        )