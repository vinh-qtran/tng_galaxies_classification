import os

import numpy as np

from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import pickle

from tqdm import tqdm


class Metadata:
    def __init__(self,
                 metadata_file : str):
        self.metadata_file = metadata_file

        self._read_metadata()

    def _read_metadata(self):
        with open(self.metadata_file,'rb') as f:
            metadata = pickle.load(f)

        for field in metadata.keys():
            setattr(self,field,metadata[field])
        self.data_fields = list(metadata.keys())

        assert 'BoxSize' in metadata.keys(), 'BoxSize not found in metadata file.'
    
    
class TNGObjects:
    def __init__(self,
                 data_file : str):
        self.data_file = data_file

        self._read_data()

    def _read_data(self):
        df = pd.read_hdf(self.data_file,key='Data')

        self.data_fields = list(df.columns)
        for field in self.data_fields:
            setattr(self,field,np.array(df[field].values))

    def _vectorize_data(self,field_initial):
        dimentions = ['.x','.y','.z']

        vectorized_data = []
        for dimention in dimentions:
            field = field_initial+dimention
            assert hasattr(self,field), f'{field} not found in data.'

            vectorized_data.append(getattr(self,field))

            delattr(self,field)
            self.data_fields.remove(field)
        self.data_fields.append(field_initial)

        setattr(self,field_initial,np.stack(vectorized_data,axis=1))

class Clusters(TNGObjects):
    default_vectorized_fields = [
        'GroupPos',
    ]

    def __init__(self,
                 data_file : str,
                 vectorized_fields : list = None):
        super().__init__(data_file)

        self.vectorized_fields = vectorized_fields if vectorized_fields is not None else self.default_vectorized_fields

        for field in self.vectorized_fields:
            self._vectorize_data(field)

class Galaxies(TNGObjects):
    default_vectorized_fields = [
        'SubhaloPos',
        'SubhaloVel',
        'SubhaloSpin',
    ]

    expand_directions = [
        np.array([i,j,k]) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]
    ]

    def __init__(self,
                 data_file : str,
                 cluster_data : Clusters,
                 box_size : float,
                 vectorized_fields : list = None):
        super().__init__(data_file)

        self.vectorized_fields = vectorized_fields if vectorized_fields is not None else self.default_vectorized_fields

        for field in self.vectorized_fields:
            self._vectorize_data(field)

        self.box_size = box_size

        self._find_parent_clusters(cluster_data)

    def _find_parent_clusters(self,cluster_data):
        expanded_GroupPos = []
        for direction in self.expand_directions:
            expanded_GroupPos.append(cluster_data.GroupPos + direction*self.box_size)
        clusters_KDTree = cKDTree(np.concatenate(expanded_GroupPos,axis=0))

        self.SubhaloParentGroupDistance, self.SubhaloParentGroupIndex = clusters_KDTree.query(self.SubhaloPos)
        self.SubhaloParentGroupIndex = self.SubhaloParentGroupIndex % cluster_data.GroupPos.shape[0]
        self.SubhaloParentGroupRadius = cluster_data.GroupRadius[self.SubhaloParentGroupIndex]
        self.SubhaloParentGroupMass = cluster_data.GroupMass[self.SubhaloParentGroupIndex]
        self.data_fields += ['SubhaloParentGroupDistance','SubhaloParentGroupIndex','SubhaloParentGroupRadius','SubhaloParentGroupMass']


class DNNDataLoader:
    def __init__(self,
                 galaxy_data : Galaxies,
                 features_transform = None,
                 boundaries : tuple = (0.5,2.0),
                 subsample_field_galaxies : bool = True,
                 train_ratio : float = 0.8,
                 batch_size : int = 4096,
                 num_workers : int = 8,
                 seed : int = 42):
        self.features = features_transform(galaxy_data) if features_transform is not None else self._default_features_transform(galaxy_data)

        self.seed = seed
        torch.manual_seed(self.seed)

        self.lower_boundary, self.upper_boundary = boundaries
        self.subsample_field_galaxies = subsample_field_galaxies
        self._galaxies_masking(galaxy_data,subsample_field_galaxies)

        self.train_ratio = train_ratio

        self.batch_size = batch_size
        self.num_workers = num_workers

        self._get_data_loaders(galaxy_data)

    def _default_features_transform(self,galaxy_data):
        Gmag = galaxy_data.SubhaloGmag
        Rmag = galaxy_data.SubhaloRmag
        Color = Gmag - Rmag # feature 1

        Mass = galaxy_data.SubhaloMass
        StellarMass = galaxy_data.SubhaloStellarMass
        MassRatio = StellarMass/Mass # feature 2

        SFR = galaxy_data.SubhaloSFR
        SSFR = SFR/StellarMass # feature 3

        GasMass = galaxy_data.SubhaloGasMass
        GasFrac = GasMass/(GasMass + StellarMass) # feature 4

        GasMetallicity = galaxy_data.SubhaloGasMetallicity # feature 5
        StarMetallicity = galaxy_data.SubhaloStarMetallicity # feature 6

        return torch.tensor(
            np.stack(
                [
                    Color,
                    np.log10(MassRatio),
                    np.log10(SSFR),
                    GasFrac,
                    GasMetallicity,
                    StarMetallicity
                ]
            ,axis=1)
        ,dtype=torch.float32)
    
    def _galaxies_masking(self,galaxy_data,subsample_field_galaxies):
        self.cluster_galaxy_mask = torch.from_numpy(
            galaxy_data.SubhaloParentGroupDistance < self.lower_boundary * galaxy_data.SubhaloParentGroupRadius
        )
        N_cluster_galaxies = torch.sum(self.cluster_galaxy_mask)

        self.full_field_galaxy_mask = torch.from_numpy(
            galaxy_data.SubhaloParentGroupDistance > self.upper_boundary * galaxy_data.SubhaloParentGroupRadius
        )
        N_field_galaxies = torch.sum(self.full_field_galaxy_mask)

        self.training_galaxy_mask = torch.logical_or(
            self.cluster_galaxy_mask,
            self.full_field_galaxy_mask
        )

        self.inference_galaxy_mask = torch.logical_not(self.training_galaxy_mask)

        if subsample_field_galaxies:
            subsample_mask = torch.rand(galaxy_data.SubhaloParentGroupDistance.shape[0]) < N_cluster_galaxies/N_field_galaxies
            self.field_galaxy_mask = torch.logical_and(
                self.full_field_galaxy_mask,
                subsample_mask
            )
        else:
            self.field_galaxy_mask = self.full_field_galaxy_mask

        print(torch.sum(self.field_galaxy_mask))

    def _get_data_loaders(self,galaxy_data):
        target = self.cluster_galaxy_mask.to(dtype=torch.int32)
        target = target[self.training_galaxy_mask]

        train_val_dataset = TensorDataset(self.features[self.training_galaxy_mask],target)
        inference_dataset = TensorDataset(
            self.features[self.inference_galaxy_mask],
            torch.tensor(galaxy_data.SubhaloParentGroupDistance[self.inference_galaxy_mask],dtype=torch.float32),
            torch.tensor(galaxy_data.SubhaloParentGroupMass[self.inference_galaxy_mask],dtype=torch.float32),
            torch.tensor(galaxy_data.SubhaloParentGroupRadius[self.inference_galaxy_mask],dtype=torch.float32),
        )

        N_data = train_val_dataset.tensors[0].shape[0]
        N_train = int(N_data*self.train_ratio)
        N_val = N_data - N_train
        print('N_train:',N_train)
        print('N_val:',N_val)

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_val_dataset,
            [N_train,N_val]
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

        self.inference_loader = DataLoader(
            inference_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import matplotlib as mpl

    mpl.style.use('classic')
    mpl.rc('xtick', labelsize=23); mpl.rc('ytick', labelsize=23)
    mpl.rc('xtick.major', size=15 , width=2)
    mpl.rc('xtick.minor', size=8, width=2, visible=True)
    mpl.rc('ytick.major', size=15 , width=2)
    mpl.rc('ytick.minor', size=8, width=2, visible=True)
    mpl.rc('lines',linewidth=3, markersize=20)
    mpl.rc('axes', linewidth=2, labelsize=30, labelpad=2.5)
    mpl.rc('legend', fontsize=25, loc='best', frameon=False, numpoints=1)

    mpl.rc('font', family='STIXGeneral')
    mpl.rc('mathtext', fontset='stix')

    def inspect_array(array,log=False,exclude_nan=False):
        '''
        Helper function to inspect the array.
        '''
        print('  Shape:', array.shape)

        array = np.log10(array) if log else array

        print('  Min:', np.nanmin(array) if exclude_nan else np.min(array))
        print('  Max:', np.nanmax(array) if exclude_nan else np.max(array))
        print('  Mean:', np.nanmean(array) if exclude_nan else np.mean(array))
        print('  Std:', np.nanstd(array) if exclude_nan else np.std(array))

    def show_galaxy_distribution(galaxies,distance_bins):
        galaxy_counts, _ = np.histogram(
            galaxies.SubhaloParentGroupDistance/galaxies.SubhaloParentGroupRadius,
            bins=distance_bins
        )

        distance_bin_centers = (distance_bins[1:] + distance_bins[:-1])/2
        bin_volumes = 4/3*np.pi*(distance_bins[1:]**3 - distance_bins[:-1]**3)

        galaxy_density = galaxy_counts/bin_volumes

        plt.figure(figsize=(10,7))
        plt.plot(distance_bin_centers,galaxy_counts)
        plt.xlabel(r'$r/R_{200}$')
        plt.ylabel('Galaxy Counts')
        plt.xscale('log')
        plt.xlim(0.1,100)
        plt.show()

        plt.figure(figsize=(10,7))
        plt.plot(distance_bin_centers,galaxy_density)
        plt.xlabel(r'$r/R_{200}$')
        plt.ylabel('Galaxy Density')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(0.1,100)
        plt.show()

    data_path = '../../data/'

    metadata = Metadata(os.path.join(data_path,'metadata.pkl'))

    clusters = Clusters(os.path.join(data_path,'group_data/reduced_data.0.h5'))
    # print('Clusters:')
    # for field in clusters.data_fields:
    #     print(f' {field}:')
    #     inspect_array(getattr(clusters,field),log=field == 'GroupMass')

    galaxies = Galaxies(os.path.join(data_path,'subhalo_data/reduced_data.0.h5'),clusters,metadata.BoxSize)
    # print('Galaxies:')
    # for field in galaxies.data_fields:
    #     print(f' {field}:')
    #     inspect_array(getattr(galaxies,field),log=field == 'SubhaloMass')
    
    # print('Galaxy Cluster Distance Histogram:')
    # show_galaxy_distribution(galaxies,distance_bins=np.linspace(0,100,1001))

    loader = DNNDataLoader(galaxies)
    # print('Features:')
    # for i,feature in enumerate(['Color','MassRatio','SSFR','GasFrac','GasMetallicity','StarMetallicity']):
    #     print(f' {feature}:')
    #     inspect_array(loader.features[:,i].cpu().numpy())

    print('Cluster Galaxy Distances:')
    inspect_array(galaxies.SubhaloParentGroupDistance[loader.cluster_galaxy_mask.cpu().numpy()]/galaxies.SubhaloParentGroupRadius[loader.cluster_galaxy_mask.cpu().numpy()])

    print('Field Galaxy Distances:')
    inspect_array(galaxies.SubhaloParentGroupDistance[loader.field_galaxy_mask.cpu().numpy()]/galaxies.SubhaloParentGroupRadius[loader.field_galaxy_mask.cpu().numpy()])

    print('Inference Galaxy Parent Cluster Masses:')
    inspect_array(galaxies.SubhaloParentGroupMass[loader.inference_galaxy_mask.cpu().numpy()])

    # print('Train DataLoader:')
    # for i, (X,Y) in enumerate(loader.train_loader):
    #     if i == 0:
    #         print(' X:')
    #         inspect_array(X.cpu().numpy())
    #         print(' Y:')
    #         inspect_array(Y.cpu().numpy())

    # print('Val DataLoader:')
    # for i, (X,Y) in enumerate(loader.val_loader):
    #     if i == 0:
    #         print(' X:')
    #         inspect_array(X.cpu().numpy())
    #         print(' Y:')
    #         inspect_array(Y.cpu().numpy())
