import os

import numpy as np
from astropy import units as u

import h5py
import pandas as pd
import pickle

from tqdm import tqdm


class FOFData:
    def __init__(self, 
                 data_dir,
                 data_fields_info,
                 maskings,
                 save_file_dir,
                 object_per_save_file,
                 metadata_fields=None):
        '''
        Initializes the FOFData object and processes the data.

        Args:
            data_dir (str): Directory containing the FOF files.
            data_fields_info (dict): Field mapping to extract specific data.
            maskings (list): List of field-value pairs for masking data.
            save_file_dir (str): Directory to save processed data.
            object_per_save_file (int): Number of objects to save per output file.
            metadata_fields (list, optional): Metadata fields to extract.
        '''
        
        self.data_dir = data_dir
        self._get_fof_files()

        self.metadata_fields = metadata_fields
        if self.metadata_fields is not None:
            self._get_and_save_metadata(self.fof_files[0])
        else:
            self.HubbleParam = 1

        self.data_fields_info = data_fields_info
        self.maskings = maskings
        self._get_data()

        self.save_file_dir = save_file_dir
        if not os.path.exists(self.save_file_dir):
            os.makedirs(self.save_file_dir)
        self.object_per_save_file = object_per_save_file
        self._save_data()


    def _get_fof_files(self):
        '''
        Identifies all FOF files in the specified directory.

        Files are identified by the prefix 'fof_subhalo_tab_'.
        '''
        self.fof_files = []
        for file in os.scandir(self.data_dir):
            if file.name.startswith('fof_subhalo_tab_'):
                self.fof_files.append(file.path)


    def _get_and_save_metadata(self,file):
        '''
        Extracts metadata from the first FOF file and saves it as a pickle file.

        Args:
            file (str): Path to the first FOF file.
        '''
        file_data = h5py.File(file, 'r')
        metadata = {}
        for field in self.metadata_fields:
            metadata[field] = file_data['Parameters'].attrs[field] if field in file_data['Parameters'].attrs else file_data['Header'].attrs[field]
            setattr(self, field, metadata[field])
        
        # print('Metadata:')
        # for field in metadata:
        #     print(f' {field}:',metadata[field])

        with open('metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)


    def _get_field_data(self,data,field_label,sub_index,h_scaling):
        '''
        Extracts data for a specific field from the HDF5 file.

        Args:
            data (h5py.Group): HDF5 group containing the data.
            field_label (str): Label of the field to extract.
            sub_index (int or None): Sub-index for multi-dimensional fields.
            h_scaling (int): Scaling factor for Hubble parameter.

        Returns:
            np.ndarray: Extracted data.
        '''
        if sub_index is not None:
            return np.array(data[field_label])[:,sub_index] * self.HubbleParam**h_scaling
        return np.array(data[field_label]) * self.HubbleParam**h_scaling
    
    def _mask_data(self,data,masking_field,masking_value):
        '''
        Masks data based on a specific field and threshold.

        Args:
            data (dict): Dictionary of data fields.
            masking_field (str): Field to use for masking.
            masking_value (float): Threshold value for masking.

        Returns:
            dict: Masked data fields.
        '''
        mask = np.logical_and(data[masking_field] > masking_value, ~np.isnan(data[masking_field]))
        for field in data:
            data[field] = data[field][mask]
        return data
        
    def _get_single_file_data(self,file):
        '''
        Processes data from a single FOF file.

        Args:
            file (str): Path to the FOF file.

        Returns:
            dict: Extracted and masked data.
        '''
        file_data = h5py.File(file, 'r')
        
        data = {}
        for field,(field_label,sub_index,h_scaling) in self.data_fields_info.items():
            data[field] = self._get_field_data(file_data,field_label,sub_index,h_scaling)

        for masking_field,masking_value in self.maskings:
            data = self._mask_data(data,masking_field,masking_value)

        return data
    
    def _get_data(self):
        '''
        Aggregates data from all FOF files.

        Concatenates data from all files into single arrays for each field.
        '''
        self.data = {field: [] for field in self.data_fields_info.keys()}

        for file in tqdm(self.fof_files):
            try:
                file_data = self._get_single_file_data(file)
                for field in self.data_fields_info.keys():
                    self.data[field].append(file_data[field])
            except KeyError:
                continue

        self.data = {field: np.concatenate(self.data[field], axis=0) for field in self.data}
        
    def _save_data(self):
        '''
        Saves processed data in HDF5 format.

        Creates multiple files, each containing a subset of the data.
        '''
        for i,start_index in tqdm(enumerate(range(0, len(self.data[self.maskings[0][0]]), self.object_per_save_file))):
            end_index = min(start_index + self.object_per_save_file, len(self.data[self.maskings[0][0]]))

            df = pd.DataFrame({
                field: self.data[field][start_index:end_index] for field in self.data_fields_info.keys()
            })

            df.to_hdf(os.path.join(self.save_file_dir,f'reduced_data.{i}.h5'), key='Data', mode='w')


class GroupData(FOFData):
    '''
    Processes group-level data from Friends-of-Friends (FOF) files and saves it in HDF5 format.

    Inherits from FOFData, specifically tailored for group-level fields such as `GroupMass`.

    Args:
        data_dir (str): Directory containing the FOF files.
        data_fields_info (dict): Field mapping to extract specific data for groups.
        masking (list, optional): List of field-value pairs for masking data. 
                                  Default is [('GroupMass', 1e3)] to filter groups based on mass.
        save_file_dir (str, optional): Directory to save processed group data. Default is 'group_data'.
        object_per_save_file (int, optional): Number of groups to save per output file. Default is 1,000,000.
        metadata_fields (list, optional): Metadata fields to extract. Default is None.

    Attributes:
        Inherits all attributes from FOFData.
    '''
    def __init__(self, 
                 data_dir,
                 data_fields_info,
                 masking=[('GroupMass',1e3)],
                 save_file_dir='group_data',
                 object_per_save_file=int(2e6),
                 metadata_fields=None):
        super().__init__(data_dir, data_fields_info, masking, save_file_dir, object_per_save_file, metadata_fields)
        

class SubHaloData(FOFData):
    '''
    Processes subhalo-level data from Friends-of-Friends (FOF) files and saves it in HDF5 format.

    Inherits from FOFData, specifically tailored for subhalo-level fields such as `SubhaloStellarMass`, `SubhaloGasMass`, and `SubhaloFlag`.

    Args:
        data_dir (str): Directory containing the FOF files.
        data_fields_info (dict): Field mapping to extract specific data for subhalos.
        masking (list, optional): List of field-value pairs for masking data.
                                  Default is:
                                  - ('SubhaloFlag', 0): Excludes flagged subhalos.
                                  - ('SubhaloStellarMass', 1e-3): Filters subhalos based on stellar mass.
        save_file_dir (str, optional): Directory to save processed subhalo data. Default is 'subhalo_data'.
        object_per_save_file (int, optional): Number of subhalos to save per output file. Default is 1,000,000.
        metadata_fields (list, optional): Metadata fields to extract. Default is None.

    Attributes:
        Inherits all attributes from FOFData.
    '''
    def __init__(self,
                 data_dir,
                 data_fields_info,
                 masking=[('SubhaloFlag',0),('SubhaloStellarMass',1e-3),],
                 save_file_dir='subhalo_data',
                 object_per_save_file=int(2e6),
                 metadata_fields=None):
        super().__init__(data_dir, data_fields_info, masking, save_file_dir, object_per_save_file, metadata_fields)


if __name__ == '__main__':
    def inspect_array(array,log=False,exclude_nan=False):
        '''
        Helper function to inspect the array.
        '''
        print(' Shape:', array.shape)

        array = np.log10(array) if log else array

        print(' Min:', np.nanmin(array) if exclude_nan else np.min(array))
        print(' Max:', np.nanmax(array) if exclude_nan else np.max(array))
        print(' Mean:', np.nanmean(array) if exclude_nan else np.mean(array))
        print(' Std:', np.nanstd(array) if exclude_nan else np.std(array))

    # Group data analysis
    group_data = GroupData(
        data_dir='groups_099',
        data_fields_info={
            'GroupMass': ('Group/Group_M_Mean200',None,-1),
            'GroupPos.x': ('Group/GroupPos',0,-1),
            'GroupPos.y': ('Group/GroupPos',1,-1),
            'GroupPos.z': ('Group/GroupPos',2,-1),
            'GroupRadius' : ('Group/Group_R_Mean200',None,-1),
        },
        metadata_fields=['UnitLength_in_cm','UnitMass_in_g','UnitVelocity_in_cm_per_s',
                         'BoxSize',
                         'HubbleParam','Omega0','OmegaLambda','Redshift','Time']
    )

    print('Group Data:')
    for field in group_data.data:
        print(field)
        if field == 'GroupMass':
            inspect_array(group_data.data[field],log=True)
        else:
            inspect_array(group_data.data[field])

    # Subhalo data analysis
    subhalo_data = SubHaloData(
        data_dir='groups_099',
        data_fields_info={
            'SubhaloFlag': ('Subhalo/SubhaloFlag',None,0),

            'SubhaloMass': ('Subhalo/SubhaloMass',None,-1),
            'SubhaloGasMass': ('Subhalo/SubhaloMassInRadType',0,-1),
            'SubhaloStellarMass': ('Subhalo/SubhaloMassInRadType',4,-1),

            'SubhaloGasMetallicity': ('Subhalo/SubhaloGasMetallicity',None,0),
            'SubhaloStarMetallicity': ('Subhalo/SubhaloStarMetallicity',None,0),

            'SubhaloSFR': ('Subhalo/SubhaloSFR',None,0),

            'SubhaloPos.x': ('Subhalo/SubhaloPos',0,-1),
            'SubhaloPos.y': ('Subhalo/SubhaloPos',1,-1),
            'SubhaloPos.z': ('Subhalo/SubhaloPos',2,-1),

            'SubhaloVel.x': ('Subhalo/SubhaloVel',0,0),
            'SubhaloVel.y': ('Subhalo/SubhaloVel',1,0),
            'SubhaloVel.z': ('Subhalo/SubhaloVel',2,0),

            'SubhaloSpin.x': ('Subhalo/SubhaloSpin',0,-1),
            'SubhaloSpin.y': ('Subhalo/SubhaloSpin',1,-1),
            'SubhaloSpin.z': ('Subhalo/SubhaloSpin',2,-1),

            'SubhaloVelDisp': ('Subhalo/SubhaloVelDisp',None,0),
            'SubhaloVmax': ('Subhalo/SubhaloVmax',None,0),

            'SubhaloUMag': ('Subhalo/SubhaloStellarPhotometrics',0,0),
            'SubhaloBMag': ('Subhalo/SubhaloStellarPhotometrics',1,0),
            'SubhaloVMag': ('Subhalo/SubhaloStellarPhotometrics',2,0),
            'SubhaloKMag': ('Subhalo/SubhaloStellarPhotometrics',3,0),
            'SubhaloGmag': ('Subhalo/SubhaloStellarPhotometrics',4,0),
            'SubhaloRmag': ('Subhalo/SubhaloStellarPhotometrics',5,0),
            'SubhaloImag': ('Subhalo/SubhaloStellarPhotometrics',6,0),
            'SubhaloZmag': ('Subhalo/SubhaloStellarPhotometrics',7,0),
        },
        metadata_fields=['UnitLength_in_cm','UnitMass_in_g','UnitVelocity_in_cm_per_s',
                         'BoxSize',
                         'HubbleParam','Omega0','OmegaLambda','Redshift','Time']
    )

    print('Subhalo Data:')
    for field in subhalo_data.data:
        print(field)
        inspect_array(subhalo_data.data[field],log=field in ['SubhaloMass','SubhaloGasMass','SubhaloStellarMass'])