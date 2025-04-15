import os
import sys
import subprocess
from pathlib import Path

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()

sys.path.append(repo_root)

##########################################################################################

import os

import numpy as np

import pickle

from tqdm import tqdm

from utils import data_reader

############################################################################################

class GalaxyStatistics:
    def __init__(self,
                 galaxy_data : data_reader.Galaxies,
                 radial_bins = np.logspace(-2,2,31),
                 min_galaxy_per_bin = 100):
        self.galaxy_data = galaxy_data

        self.radial_bins = radial_bins
        self.min_galaxy_per_bin = min_galaxy_per_bin

        self.radial_bin_indices = np.digitize(galaxy_data.SubhaloParentGroupDistance/galaxy_data.SubhaloParentGroupRadius, self.radial_bins) - 1

    def get_feature_profile(self,feature_func):
        feature_data = feature_func(self.galaxy_data)

        radial_bin_centers = (self.radial_bins[1:] + self.radial_bins[:-1]) / 2
        feature_profile = []
        feature_profile_err = []

        idx_modifier = 0
        for i in range(len(self.radial_bins)-1):
            mask = self.radial_bin_indices == i

            if np.sum(mask) < self.min_galaxy_per_bin:
                radial_bin_centers = np.delete(radial_bin_centers, i-idx_modifier)
                idx_modifier += 1
                continue

            feature_profile.append(np.mean(feature_data[mask]))
            feature_profile_err.append(np.std(feature_data[mask]))

        return radial_bin_centers, np.array(feature_profile), np.array(feature_profile_err)
    
    def get_feature_histogram(self,feature_func,feature_bins):
        feature_data = feature_func(self.galaxy_data)

        feature_hist, _, _ = np.histogram2d(
            self.galaxy_data.SubhaloParentGroupDistance/self.galaxy_data.SubhaloParentGroupRadius,
            feature_data,
            bins=[self.radial_bins,feature_bins],
        )
        radial_bin_centers = (self.radial_bins[1:] + self.radial_bins[:-1]) / 2
        feature_bin_centers = (feature_bins[1:] + feature_bins[:-1]) / 2
        feature_hist = feature_hist.T
        
        return radial_bin_centers, feature_bin_centers, feature_hist