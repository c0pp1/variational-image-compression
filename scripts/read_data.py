import numpy as np


def read_data_numpy(data_path, format='channels_last') -> np.ndarray:
    """Read data from file and return a numpy array."""
    with open(data_path, 'rb') as f:
        # read data from file
        X = np.fromfile(f, dtype=np.uint8)
        # reshape data to (num_images, 3, 96, 96)
        X = np.reshape(X, (-1, 3, 96, 96))
        # transpose data to (num_images, 96, 96, 3) to match the image format
        if format=='channels_last':
            X = np.transpose(X, (0, 3, 2, 1))
            return X
        elif format=='channels_first':
            X = np.transpose(X, (0, 1, 3, 2))
            return X
        else :
            raise ValueError('format must be either "channels_last" or "channels_first"')