import numpy as np

from scipy.spatial import KDTree

import pandas as pd
import pickle

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