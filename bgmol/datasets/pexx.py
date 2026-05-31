import os
import numpy as np
import pickle
import lzma
from bgmol.datasets.base import DataSet
from bgmol.api import system_by_name
# from bgmol.tpl.hdf5 import HDF5TrajectoryFile, load_hdf5

__all__ = ["PolyEthyleneData",  # generic configurations dataset
           "PolyEthylene_fx24",  # npt md configurations dataset for pe50, flexible bonds
        ]

def read_pickle_lzma(obj, n_frames=None, stride=None, atom_indices=None):
    """Read the pickle.lzma archived where a dict with the
    coordinates and the energy per chain configuration.
    """
    with lzma.open(obj.trajectory_file, "rb") as f:
        data = pickle.load(f)
    mw = data['frames'].shape[1]
    assert mw == obj._mw, f"ERROR: configurations in {obj.trajectory_file} file have mw = {mw} different than {obj._mw}"
    data['frames'] /= 10.0 # convert to nanometers
    for name in data['thermo'].dtype.names:
        data['thermo'][name] *= 4.184 # convert to kJ/mol
    obj.data = data

    # center around com positioned at zero
    # j0 = int(obj._mw//2)-1
    _xyz = data['frames']
    cm  = _xyz.sum(axis=1)/obj._mw
    centered = _xyz - cm[:, None, :]
    inertia = centered.transpose(0, 2, 1) @ centered
    _, eigenvectors = np.linalg.eigh(inertia)
    _xyz[:,:,:] = centered @ eigenvectors
    # for i in range(_xyz.shape[0]):
    #     # _xyz[i,:,:] -= _xyz[i,j0,:]
    #     # Center the molecule (COM)
    #     centered = _xyz[i,:,:] - cm[i,:]
    #     # Compute Inertia Tensor (simplified with equal weights)
    #     inertia = np.dot(centered.T, centered)
    #     # Get Principal Axes
    #     _, eigenvectors = np.linalg.eigh(inertia)
    #     # Rotate to align
    #     _xyz[i,:,:] = np.dot(centered, eigenvectors)
    obj._xyz = _xyz

    obj._thermo = data['thermo'] 
    obj._energies = obj._thermo['etot'].copy() 

class PolyEthyleneData(DataSet):
    """PolyEthylene basic class"
    """
    url = "xxx"
    md5 = "xxx"
    num_frames = 0
    size = 0  # in bytes
    selection = "all"
    openmm_version = "x.x.x"
    date = "2024/05"
    author = "Loukas Peristeras"

    def __init__(self, mw, filename, first=0, nrigid=3, root=os.getcwd(), download: bool = False, read: bool = False):
        self._mw = mw
        self._filename = filename
        super(PolyEthyleneData, self).__init__( 
            root=root, download=download, read=read)
        self._system = system_by_name("PolyEthylene", mw=self._mw, nrigid=nrigid, first=first, root=root)
        self._temperature = 450

    @property
    def trajectory_file(self):
        return os.path.join(self.root, self._filename)

    def read(self, n_frames=None, stride=None, atom_indices=None):
        read_pickle_lzma(self, n_frames, stride, atom_indices)

class PolyEthylene_fx24(PolyEthyleneData):
    """Example of specific implementation of PolyEthylene class.
       Provides access to data for lpem configurations for pe24, 
       flexible bonds,  melt at 450 K, 1 atm.
    """
    url = "xxx"
    md5 = "xxx"
    num_frames = 0
    size = 0  # in bytes
    selection = "all"
    openmm_version = "x.x.x"
    date = "2024/05"
    author = "Loukas Peristeras"

    def __init__(self, first=0, nrigid=3, root=os.getcwd(), download: bool = False, read: bool = False):
        super(PolyEthylene_fx24, self).__init__( 24, "npt_1000200x24f.pickle.lzma",
            root=root, download=download, read=read)

