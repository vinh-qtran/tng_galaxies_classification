import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

class ClusterInference:
    def __init__(self,
                 test_loader,
                 model,model_state_dict,
                 mass_bins,radial_bins,
                 min_galaxy_per_bin=10,
                 device='mps'):
        self.test_loader = test_loader

        self.device = device

        self.model = model
        self.model.load_state_dict(model_state_dict)
        self.model.to(device)

        self._classify_galaxies()

        self.mass_bins = mass_bins
        self.radial_bins = radial_bins

        self.min_galaxy_per_bin = min_galaxy_per_bin

    def _classify_galaxies(self):
        galaxy_cluster_masses = []
        galaxy_distances = []
        galaxy_labels = []

        with torch.no_grad():
            for X, _, cluster_mass, distance in tqdm(self.test_loader):
                X = X.to(self.device)
                lable = self.model(X).flatten()

                galaxy_cluster_masses += cluster_mass.tolist()
                galaxy_distances += distance.tolist()
                galaxy_labels += lable.tolist()

        self.galaxy_cluster_masses = np.array(galaxy_cluster_masses)
        self.galaxy_distances = np.array(galaxy_distances)
        self.galaxy_labels = np.array(galaxy_labels)
    
    def _get_bin_radial_labels(self,galaxy_distances,galaxy_labels):
        galaxy_bin_indices =np.digitize(galaxy_distances, self.radial_bins) - 1
        
        radial_bin_centers = (self.radial_bins[1:] + self.radial_bins[:-1]) / 2

        radial_labels = np.array([])
        radial_labels_err = np.array([])

        idx_modifier = 0
        for i in range(len(self.radial_bins) - 1):
            mask = galaxy_bin_indices == i
            if np.sum(mask) < self.min_galaxy_per_bin:
                radial_bin_centers = np.delete(radial_bin_centers, i-idx_modifier)
                idx_modifier += 1
                continue

            radial_labels = np.append(radial_labels, np.median(galaxy_labels[mask]))
            radial_labels_err = np.append(radial_labels_err, np.median(np.abs(galaxy_labels[mask]-np.median(galaxy_labels[mask])))/0.6745)

        return radial_bin_centers, radial_labels, radial_labels_err
    
    def _get_single_boundary(self, radial_bin_centers, radial_labels, threshold):
        cross_indices = np.where(np.diff(np.sign(radial_labels - threshold)))[0]
        if not len(cross_indices):
            return np.nan
        
        first_cross_idx = cross_indices[0]
        
        return np.interp(threshold,[radial_labels[first_cross_idx+1], radial_labels[first_cross_idx]], 
                                   [radial_bin_centers[first_cross_idx+1], radial_bin_centers[first_cross_idx]])
            
    def get_boundaries(self,sigmoid=False,n_MC=1000):
        galaxy_cluster_bin_indices = np.digitize(self.galaxy_cluster_masses, self.mass_bins) - 1
        
        mass_bin_centers = (self.mass_bins[1:] + self.mass_bins[:-1]) / 2

        radial_boundaries = np.array([])
        radial_boundaries_err = np.array([])

        threshold = 0.5 if sigmoid else 0.0

        idx_modifier = 0
        for i in range(self.mass_bins.shape[0] - 1):
            mask = galaxy_cluster_bin_indices == i

            if np.sum(mask) < self.min_galaxy_per_bin*len(self.radial_bins):
                mass_bin_centers = np.delete(mass_bin_centers, i-idx_modifier)
                idx_modifier += 1
                continue

            radial_bin_centers, radial_labels, radial_labels_err = self._get_bin_radial_labels(self.galaxy_distances[mask],self.galaxy_labels[mask])
            # print('mass_bin_centers:',np.log10(mass_bin_centers[i-idx_modifier]))
            # print('radial_labels:',radial_labels)
            # print('radial_labels_err:',radial_labels_err)
            
            MC_radial_boundaries = []

            for j in range(n_MC):
                sampled_radial_labels = np.random.normal(radial_labels,radial_labels_err)
                MC_radial_boundaries.append(self._get_single_boundary(radial_bin_centers, sampled_radial_labels, threshold))

            radial_boundaries = np.append(radial_boundaries, np.nanmedian(MC_radial_boundaries))
            radial_boundaries_err = np.append(radial_boundaries_err, np.nanmedian(np.abs(MC_radial_boundaries-np.nanmedian(MC_radial_boundaries)))/0.6745)

        return mass_bin_centers, radial_boundaries, radial_boundaries_err

           
        