import h5py
import torch
import numpy as np

def create_hdf5_generator(file_path, batch_size):
    file = h5py.File(file_path)
    data_size = file['data'].shape[0]

    while True: # loop through the dataset indefinitely
        for i in np.arange(0, data_size, batch_size):
            data = file['data'][i:i+batch_size]
            labels = file['labels'][i:i+batch_size]
            # converting data into torch format
            data  = torch.from_numpy(data)
            # converting the lables into torch format
            labels = torch.from_numpy(labels)
            yield data, labels



    