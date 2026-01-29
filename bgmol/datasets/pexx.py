import os
import numpy as np
import pickle
import lzma
from bgmol.datasets.base import DataSet
from bgmol.api import system_by_name
# from bgmol.tpl.hdf5 import HDF5TrajectoryFile, load_hdf5

__all__ = ["PolyEthylene_fx24",  # npt md configurations for pe24, flexible bonds
           "PolyEthylene_fx50",  # npt md configurations for pe50, flexible bonds
           "PolyEthylene_fx74",  # npt md configurations for pe75, flexible bonds
           ]

def read_pickle_lzma(obj, n_frames=None, stride=None, atom_indices=None):
    """Read the pickle.lzma archived where a dict with the
    coordinates and the energy per chain configuration.
    """
    with lzma.open(obj.trajectory_file, "rb") as f:
        data = pickle.load(f)
    assert data['frames'].shape[1] == obj._mw
    data['frames'] /= 10.0 # convert to nanometers
    for name in data['thermo'].dtype.names:
        data['thermo'][name] *= 4.184 # convert to kJ/mol
    obj.data = data

    # center around com positioned at zero
    # j0 = int(obj._mw//2)-1
    _xyz = data['frames']
    cm  = _xyz.sum(axis=1)/obj._mw
    for i in range(_xyz.shape[0]):
        _xyz[i,:,:] -= cm[i,:]
        # _xyz[i,:,:] -= _xyz[i,j0,:]
    obj._xyz = _xyz

    obj._thermo = data['thermo'] 
    obj._energies = obj._thermo['etot'].copy() 


class PolyEthylene_fx24(DataSet):
    """PolyEthylene melt at 450 K, 1 atm.
    Lpem configurations for pe25, flexible bonds
    """
    url = "xxx"
    md5 = "xxx"
    num_frames = 1000000
    size = 0  # in bytes
    selection = "all"
    openmm_version = "x.x.x"
    date = "2024/05"
    author = "Loukas Peristeras"

    def __init__(self, first=0, nrigid=3, root=os.getcwd(), download: bool = False, read: bool = False):
        self._mw = 24
        super(PolyEthylene_fx24, self).__init__( 
            root=root, download=download, read=read)
        self._system = system_by_name("PolyEthylene", mw=self._mw, nrigid=nrigid, first=first, root=root)
        self._temperature = 450

    @property
    def trajectory_file(self):
        return os.path.join(self.root, "npt_1000200x24f.pickle.lzma")

    def read(self, n_frames=None, stride=None, atom_indices=None):
        read_pickle_lzma(self, n_frames, stride, atom_indices)


class PolyEthylene_fx50(DataSet):
    """PolyEthylene melt at 450 K, 1 atm.
    Lpem configurations for pe50, flexible bonds
    """
    url = "xxx"
    md5 = "xxx"
    num_frames = 1000000
    size = 0  # in bytes
    selection = "all"
    openmm_version = "x.x.x"
    date = "2024/05"
    author = "Loukas Peristeras"

    def __init__(self, first=0, nrigid=3, root=os.getcwd(), download: bool = False, read: bool = False):
        self._mw = 50
        super(PolyEthylene_fx50, self).__init__(
            root=root, download=download, read=read)
        self._system = system_by_name("PolyEthylene", mw=self._mw, first=first, nrigid=nrigid, root=root)
        self._temperature = 450

    @property
    def trajectory_file(self):
        return os.path.join(self.root, "npt_1000200x50f.pickle.lzma")

    def read(self, n_frames=None, stride=None, atom_indices=None):
        read_pickle_lzma(self, n_frames, stride, atom_indices)


class PolyEthylene_fx74(DataSet):
    """PolyEthylene melt at 450 K, 1 atm.
    Lpem configurations for pe25, flexible bonds, relaxed with 10ps NVT
    """
    url = "xxx"
    md5 = "xxx"
    num_frames = 1000000
    size = 0  # in bytes
    selection = "all"
    openmm_version = "x.x.x"
    date = "2024/05"
    author = "Loukas Peristeras"

    def __init__(self, first=0, nrigid=3, root=os.getcwd(), download: bool = False, read: bool = False):
        self._mw = 74
        super(PolyEthylene_fx74, self).__init__(
            root=root, download=download, read=read)
        self._system = system_by_name("PolyEthylene", mw=self._mw, nrigid=nrigid, first=first, root=root)
        self._temperature = 450

    @property
    def trajectory_file(self):
        return os.path.join(self.root, "npt_1000200x74f.pickle.lzma")

    def read(self, n_frames=None, stride=None, atom_indices=None):
        read_pickle_lzma(self, n_frames, stride, atom_indices)
