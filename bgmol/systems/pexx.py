from openmm.unit import *
from openmm.app import *
from openmm import *
# from ..tpl.download import download_url
from .base import OpenMMSystem
import mdtraj as md
from bgmol.util.importing import import_openmm
import os
import tempfile
import numpy as np

import sys
__pymol_path = "/Users/loukas.peristeras/work/demokritos/projects/natasa/bgmol"
if not __pymol_path in sys.path:
    sys.path.append(__pymol_path)
__pgflow_path = "/Users/loukas.peristeras/work/demokritos/projects/natasa/bgflow"
if not __pgflow_path in sys.path:
    sys.path.append(__pgflow_path)

_, unit, app = import_openmm()


__all__ = ["PolyEthylene"]

# define Trappe elements
# NOTE: this causes problems with mdtraj because it does not support user-defined elements
_uaElement_ch4 = app.Element(
    number=1000, name='UA-CH4', symbol='_CH4', mass=16.04300)
_uaElement_ch3 = app.Element(
    number=1000, name='UA-CH3', symbol='_CH3', mass=15.03500)
_uaElement_ch2 = app.Element(
    number=1000, name='UA-CH2', symbol='_CH2', mass=14.02700)


class PolyEthylene(OpenMMSystem):
    """Polyethylene (PE) melt with 25 carbon atoms in chain's backbone,
    TODO provide a ref e.g., Peristeras et al. (2024)

    Notes
    -----
    Currently, it is only locally available and the root argument in the
    initializer should be provided.
    """

    # LDP consider to upload it in a repository:
    # Requires an internet connection to download the initial structure.
    # url = "http://ftp.mi.fu-berlin.de/pub/cmb-data/bgmol/systems/ala2/"

    def __init__(self,  mw=0, first=0, nrigid=3, flex=True, root=tempfile.gettempdir(), download=True):
        assert mw > 1, 'ERROR: check mw in PolyEthylene.__init__'

        super(PolyEthylene, self).__init__()
        self._mw = mw
        self._flex = flex
        self._first = first
        self._nrigid = nrigid

        # download gro file
        filename = "pe{mw}.gro".format(mw=mw)
        full_filename = os.path.join(root, filename)
        # TODO upload the files in a repository
        # if download:
        #     download_url(self.url + filename, root, filename, md5="xxx")
        assert os.path.isfile(full_filename)
        gro = GromacsGroFile(full_filename)

        # download force field file
        filename = "trappe-ua.xml"
        ff_full_filename = os.path.join(root, filename)
        # TODO upload the files in a repository
        # if download:
        #     download_url(self.url + filename, root, filename, md5="xxx")
        assert os.path.isfile(ff_full_filename)
        # all the the elements used in the force field file they should have
        # been registered by now (i.e., _uaElement_ch*)
        ff = app.ForceField(ff_full_filename)

        self._system = ff.createSystem(
            PolyEthylene.get_topology(mw),
            removeCMMotion=False,
            # nonbondedMethod=app.NoCutoff,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.2 * nanometer,
            constraints=None if flex else app.AllBonds
        )
        self._positions = gro.getPositions(
            asNumpy=True).value_in_unit(nanometer)
        self._topology = PolyEthylene.get_topology(mw)
        self.rigid_block = PolyEthylene.get_rigid_block(
            mw, first=self._first, nrigid=self._nrigid)
        self.z_matrix = PolyEthylene.get_z_matrix(
            mw, first=self._first, nrigid=self._nrigid)

    @staticmethod
    def get_topology(mw):
        topology = Topology()

        # Create the chain
        chain = topology.addChain()

        # Create names and elements for the atoms
        _nmidle = mw-2
        _names = ["CA"] + ["CT"]*_nmidle + ["CA"]
        # NOTE: use a std element (C) to avoid problems during the map of the openmm
        # to mdtraj topology structure. This is also affect the corresponding xml force
        # field file trappe-ua.xml.
        _elements = [app.Element.getBySymbol('C')]*mw
        # _elements = [_uaElement_ch3] + [_uaElement_ch2]*_nmidle + [_uaElement_ch3]

        # Create the first residue/atom
        residue = topology.addResidue(name=_names[0], chain=chain)
        atom1 = topology.addAtom(
            name=_names[0], element=_elements[0], residue=residue)
        # Create the rest of residues/atoms and bonds
        # for i, _el in enumerate(_elements, start=2):
        #     atom2 = topology.addAtom(name="C{i}".format(i=i), element=_el, residue=residue)
        for _nm, _el in zip(_names[1:], _elements[1:]):
            residue = topology.addResidue(name=_nm, chain=chain)
            atom2 = topology.addAtom(name=_nm, element=_el, residue=residue)
            topology.addBond(atom1, atom2)
            atom1 = atom2

        return topology

    @staticmethod
    def get_global_z_matrix(mw, first=0, nrigid=3):

        RIGID_BLOCK = PolyEthylene.get_rigid_block(mw, first, nrigid)

        # append the rigid block part
        GLOBAL_Z_MATRIX = np.row_stack((
            PolyEthylene.get_z_matrix(mw, first),
            np.array([
                [RIGID_BLOCK[2], RIGID_BLOCK[1], RIGID_BLOCK[0], -1],
                [RIGID_BLOCK[0], RIGID_BLOCK[1],             -1, -1],
                [RIGID_BLOCK[1],             -1,             -1, -1]])
        ))
        return GLOBAL_Z_MATRIX

    @staticmethod
    def get_z_matrix(mw, first=0, nrigid=3):

        RIGID_BLOCK = PolyEthylene.get_rigid_block(mw, first, nrigid)

        # initialize with the segment left to the rigid block
        Z_MATRIX = [[i, i+1, i+2, i+3]
                    for i in range(RIGID_BLOCK[0]-1, -1, -1)] if RIGID_BLOCK[0] > 0 else []

        # if needed, prepend the segment right to the rigid block
        Z_MATRIX += [[i, i-1, i-2, i-3]
                     for i in range(RIGID_BLOCK[-1]+1, mw)]

        return np.array(Z_MATRIX)

    @staticmethod
    def get_rigid_block(mw, first=0, nrigid=3):
        return np.arange(first, first+nrigid)

    def compute_bonds(self, traj):
        """Compute backbone bond probability.

        Parameters
        ----------
        traj : mdtraj.Trajectory
        """
        bonds_atoms = [[i-1, i] for i in range(1, self._mw)]
        return md.compute_distances(traj, atom_pairs=bonds_atoms).flatten()

    def compute_theta(self, traj):
        """Compute backbone bond angles probability.

        Parameters
        ----------
        traj : mdtraj.Trajectory
        """
        theta_atoms = [[i-2, i-1, i] for i in range(2, self._mw)]
        return md.compute_angles(traj, angle_indices=theta_atoms).flatten()

    def compute_phi(self, traj):
        """Compute backbone dihedrals probability.

        Parameters
        ----------
        traj : mdtraj.Trajectory
        """
        phi_atoms = [[i-3, i-2, i-1, i] for i in range(3, self._mw)]
        return md.compute_dihedrals(traj, indices=phi_atoms).flatten()

    def compute_phi_psi(self, traj):
        """Compute backbone dihedrals sequential pairs joint probability.

        Parameters
        ----------
        traj : mdtraj.Trajectory
        """
        phi_atoms = [[i-3, i-2, i-1, i] for i in range(3, self._mw)]
        phi = md.compute_dihedrals(traj, indices=phi_atoms[:-1]).flatten()
        psi = md.compute_dihedrals(traj, indices=phi_atoms[1:]).flatten()
        return phi, psi

    def compute_energy(self, traj):
        """Compute energy.

        Parameters
        ----------
        traj : mdtraj.Trajectory
        """
        system = self._system
        topology = PolyEthylene.get_topology(self._mw)

        os.environ['MAGICK_OCL_DEVICE'] = 'OFF'

        for i, f in enumerate(system.getForces()):
            f.setForceGroup(i)

        integrator = LangevinMiddleIntegrator(
            450*kelvin, 1/picosecond, 0.0001*picoseconds)
        simulation = Simulation(topology, system, integrator)
        for iframe in range(2):
            simulation.context.setPositions(traj.openmm_positions(iframe))
            for i, f in enumerate(system.getForces()):
                state = simulation.context.getState(getEnergy=True, groups={i})
                print(f.getName(), state.getPotentialEnergy())
            print("="*20)
