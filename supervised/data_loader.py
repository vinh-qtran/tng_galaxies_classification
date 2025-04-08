import os

import numpy as np

from scipy.spatial import KDTree

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

        setattr(self,field_initial,np.stack(vectorized_data,axis=1))
        self.data_fields.append(field_initial)

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
        clusters_KDTree = KDTree(cluster_data.GroupPos,boxsize=self.box_size)

        self.SubhaloParentGroupDistance, self.SubhaloParentGroupIndex = clusters_KDTree.query(self.SubhaloPos)
        self.SubhaloParentGroupIndex = self.SubhaloParentGroupIndex.astype(int)
        self.SubhaloParentGroupRadius = cluster_data.GroupRadius[self.SubhaloParentGroupIndex]
        self.SubhaloParentGroupMass = cluster_data.GroupMass[self.SubhaloParentGroupIndex]
        self.data_fields += ['SubhaloParentGroupDistance','SubhaloParentGroupIndex','SubhaloParentGroupRadius','SubhaloParentGroupMass']


class GalaxyDataLoader:
    def __init__(self,
                 galaxy_data : Galaxies,
                 features_transform = None,
                 boundaries : tuple = (1,1),
                 trainval_ratio : float = 0.9,
                 train_ratio : float = 0.8,
                 seed : int = 42):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(self.seed)

        self.galaxy_data = galaxy_data
        self.features_transform = features_transform if features_transform is not None else self._default_features_transform

        self.lower_boundary, self.upper_boundary = boundaries
        self.cluster_galaxy_mask, self.field_galaxy_mask = self._galaxies_masking()

        self.trainval_ratio = trainval_ratio
        self.train_ratio = train_ratio

        self.train_galaxy_mask, self.val_galaxy_mask, self.test_galaxy_mask = self._get_train_val_test_masks()

    def _default_features_transform(self):
        Gmag = self.galaxy_data.SubhaloGmag
        Rmag = self.galaxy_data.SubhaloRmag
        Color = Gmag - Rmag # feature 1

        Mass = self.galaxy_data.SubhaloMass
        StellarMass = self.galaxy_data.SubhaloStellarMass
        MassRatio = StellarMass/Mass # feature 2

        SFR = self.galaxy_data.SubhaloSFR
        SSFR = SFR/StellarMass # feature 3
        zero_SSFR_mask = SSFR <= 0
        SSFR[zero_SSFR_mask] = 1e-16

        GasMass = self.galaxy_data.SubhaloGasMass
        GasFrac = GasMass/(GasMass + StellarMass) # feature 4

        GasMetallicity = self.galaxy_data.SubhaloGasMetallicity # feature 5
        StarMetallicity = self.galaxy_data.SubhaloStarMetallicity # feature 6

        return np.stack(
            [
                Color,
                np.log10(MassRatio),
                np.log10(SSFR),
                GasFrac,
                GasMetallicity,
                StarMetallicity
            ]
        ,axis=1)
    
    def _subsample_galaxies(self,galaxy_mask,subsampling_fraction):
        subsample_mask = torch.rand(galaxy_mask.shape[0]) < subsampling_fraction

        return torch.logical_and(
            galaxy_mask,
            subsample_mask
        )
    
    def _galaxies_masking(self):
        cluster_galaxy_mask = torch.from_numpy(np.logical_and(
            self.galaxy_data.SubhaloParentGroupDistance < self.lower_boundary * self.galaxy_data.SubhaloParentGroupRadius,
            self.galaxy_data.SubhaloParentGroupDistance > 0
        ))
        N_cluster_galaxies = torch.sum(cluster_galaxy_mask)
        print('Initial N_cluster_galaxies:',int(N_cluster_galaxies))

        field_galaxy_mask = torch.from_numpy(
            self.galaxy_data.SubhaloParentGroupDistance > self.upper_boundary * self.galaxy_data.SubhaloParentGroupRadius
        )
        N_field_galaxies = torch.sum(field_galaxy_mask)
        print('Initial N_field_galaxies:',int(N_field_galaxies))

        if N_cluster_galaxies > N_field_galaxies:
            cluster_galaxy_mask = self._subsample_galaxies(cluster_galaxy_mask,N_field_galaxies/N_cluster_galaxies)
        else:
            field_galaxy_mask = self._subsample_galaxies(field_galaxy_mask,N_cluster_galaxies/N_field_galaxies)

        print('Subsampled N_cluster_galaxies:',int(torch.sum(cluster_galaxy_mask)))
        print('Subsampled N_field_galaxies:',int(torch.sum(field_galaxy_mask)))

        return cluster_galaxy_mask, field_galaxy_mask

    def _get_train_val_test_masks(self):
        # Get the masks for the training, validation and test sets (ensuring no cross-contamination between clusters)
        all_cluster_indices = np.unique(self.galaxy_data.SubhaloParentGroupIndex)
        N_cluster = len(all_cluster_indices)
        
        trainval_cluster_indices = np.random.choice(
            all_cluster_indices,
            size=int(N_cluster*self.trainval_ratio),
            replace=False
        )
        test_cluster_indices = np.setdiff1d(all_cluster_indices,trainval_cluster_indices)
        train_cluster_indices = np.random.choice(
            trainval_cluster_indices,
            size=int(len(trainval_cluster_indices)*self.train_ratio),
            replace=False
        )
        val_cluster_indices = np.setdiff1d(trainval_cluster_indices,train_cluster_indices)

        print('N_train_cluster:',len(train_cluster_indices))
        print('N_val_cluster:',len(val_cluster_indices))
        print('N_test_cluster:',len(test_cluster_indices))

        trainval_galaxy_mask = np.logical_or(
            self.cluster_galaxy_mask,
            self.field_galaxy_mask
        )

        train_galaxy_mask = np.logical_and(
            np.isin(
                self.galaxy_data.SubhaloParentGroupIndex,
                train_cluster_indices
            ),
            trainval_galaxy_mask
        )
        val_galaxy_mask = np.logical_and(
            np.isin(
                self.galaxy_data.SubhaloParentGroupIndex,
                val_cluster_indices
            ),
            trainval_galaxy_mask
        )
        test_galaxy_mask = np.isin(
            self.galaxy_data.SubhaloParentGroupIndex,
            test_cluster_indices
        )

        return train_galaxy_mask, val_galaxy_mask, test_galaxy_mask
    
    def get_data_loaders(self,batch_size=2048,num_workers=8):
        # Get features and targets
        features = self.features_transform(self.galaxy_data)
        targets = np.zeros(self.galaxy_data.SubhaloParentGroupIndex.shape[0])
        targets[self.cluster_galaxy_mask] = 1

        # To DataLoader
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(features[self.train_galaxy_mask]),
                torch.tensor(targets[self.train_galaxy_mask],dtype=torch.int32)
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(features[self.val_galaxy_mask]),
                torch.tensor(targets[self.val_galaxy_mask],dtype=torch.int32)
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(features[self.test_galaxy_mask]),
                torch.from_numpy(self.galaxy_data.SubhaloParentGroupDistance[self.test_galaxy_mask]),
                torch.from_numpy(self.galaxy_data.SubhaloParentGroupRadius[self.test_galaxy_mask]),
                torch.from_numpy(self.galaxy_data.SubhaloParentGroupMass[self.test_galaxy_mask]),
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )

        return train_loader, val_loader, test_loader
    

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
        plt.ylabel('Galaxy Number Density')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(0.1,100)
        plt.show()

    data_path = '../../data/'

    metadata = Metadata(os.path.join(data_path,'metadata.pkl'))
    print('Metadata:')
    for field in metadata.data_fields:
        print(f' {field}:')
        print(getattr(metadata,field))

    clusters = Clusters(os.path.join(data_path,'group_data/reduced_data.0.h5'))
    print('Clusters:')
    for field in clusters.data_fields:
        print(f' {field}:')
        inspect_array(getattr(clusters,field),log=field == 'GroupMass')

    galaxies = Galaxies(os.path.join(data_path,'subhalo_data/reduced_data.0.h5'),clusters,metadata.BoxSize/metadata.HubbleParam)
    print('Galaxies:')
    for field in galaxies.data_fields:
        print(f' {field}:')
        inspect_array(getattr(galaxies,field),log=field in ['SubhaloMass','SubhaloStellarMass','SubhaloGasMass','SubhaloParentGroupMass'])
    
    print('Galaxy Cluster Distance Histogram:')
    show_galaxy_distribution(galaxies,distance_bins=np.linspace(0,100,1001))

    loader = GalaxyDataLoader(galaxies)
    # print('Features:')
    # for i,feature in enumerate(['Color','MassRatio','SSFR','GasFrac','GasMetallicity','StarMetallicity']):
    #     print(f' {feature}:')
    #     inspect_array(loader.features[:,i].cpu().numpy())

    # print('Galaxies Distances:')
    # inspect_array(galaxies.SubhaloParentGroupDistance/galaxies.SubhaloParentGroupRadius)

    train_loader, val_loader, test_loader = loader.get_data_loaders(batch_size=2048,num_workers=8)

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
