import numpy as np
import h5py
import dotenv

def get_coords(filename : str) -> np.ndarray:
    
    ds_arr = None

    with h5py.File(filename, "r") as f:
        a_group_key = list(f.keys())[0]
        ds_arr = f[a_group_key][()] 

    return ds_arr
