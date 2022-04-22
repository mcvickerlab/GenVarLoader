import h5py
import numpy as np


def load_onehot_h5(in_h5, chrom_list=None):
    """Load onehot encoded genome from H5 to dictionary

    :param in_h5: onehot encoded H5 made by 'writefasta' -e command
    :type in_h5: str
    :param chrom_list: Chromosomes to encode, defaults to ALL chroms
    :type chrom_list: list of str, optional
    :return: Dictionary with keys: [chrom] and one-hot encoded data
    :rtype: dict of np.ndarray
    """
    
    if not h5py.is_hdf5(in_h5):
        raise ValueError("File is not valid HDF5")
    
    onehot_dict = {}
    with h5py.File(in_h5, "r") as file:
        
        if "encode_spec" in file.attrs:
            print(f"Encode Spec: {file.attrs['encode_spec']}")
        
        if not chrom_list:
            chrom_list = list(file.keys())
        
        onehot_dict = {chrom:file[chrom]["onehot"][:] for chrom in chrom_list}

    return onehot_dict


def load_depth_h5(in_h5, chrom_list=None):
    """Load read depths from H5 to dictionary

    :param in_h5: depth H5 made from 'writedepth' command
    :type in_h5: str
    :param chrom_list: Chromosomes to encode, defaults to ALL chroms
    :type chrom_list: list of str, optional
    :return: Dictionary with keys: [chrom] and array(0-based)
             with read depth per pos
    :rtype: dict of np.ndarray
    """
    
    if not h5py.is_hdf5(in_h5):
        raise ValueError("File is not valid HDF5")
    
    with h5py.File(in_h5, "r") as file:

        if not chrom_list:
            chrom_list = list(file.keys())

        depth_dict = {chrom:file[chrom]["depth"][:] for chrom in chrom_list}

    return depth_dict


def load_coverage_h5(in_h5, chrom_list=None):
    """Load allele coverage from H5 to dictionary

    :param in_h5: coverage H5 made from 'writecoverage' command
    :type in_h5: str
    :param chrom_list: Chromosomes to encode, defaults to ALL chroms
    :type chrom_list: list of str, optional
    :return: Dictionary with keys: [chrom] and 4xN matrix with row order A, C, G, T 
             containing allelic coverage per pos(0-based)
    :rtype: dict of np.ndarray
    """
    
    if not h5py.is_hdf5(in_h5):
        raise ValueError("File is not valid HDF5")
    
    with h5py.File(in_h5, "r") as file:

        if not chrom_list:
            chrom_list = list(file.keys())

        coverage_dict = {chrom:file[chrom]["coverage"][:] for chrom in chrom_list}

    return coverage_dict
