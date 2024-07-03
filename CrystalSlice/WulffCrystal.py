import numpy as np
import matplotlib.pyplot as plt
from .Cuboid import Cuboid
import open3d
import json
from ase.spacegroup import crystal
from wulffpack import SingleCrystal
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.analysis.wulff import WulffShape
from .base import Custom, sort_ascend, get_connections

def s_i_l_from_wulff(crystal):
    """ Approximate S, I and L for wulff construct object. Assume oriented bounding box

    Args:
        crystal: pymatgen object

    Returns:
        S, I, L (float): approximate parameters calculated
    """
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(crystal.wulff_convex.points)
    bb = open3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    bb_corners = np.dot(bb.get_box_points(), bb.R)

    maxs = np.max(bb_corners, axis = 0)
    mins = np.min(bb_corners, axis = 0)
    dimensions = maxs - mins
    sorted_dims = np.sort(dimensions)

    return sorted_dims[0], sorted_dims[1], sorted_dims[2]

def axis_align_s_i_l(crystal):
    """calculate axis aligned bounding box S, I and L to relate to cuboid

    Args:
        corner_points (list/ndarray): list of corner coordinates

    Returns:
        S, I, L (float): dimensions of bounding box
    """
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(crystal.wulff_convex.points)
    bb = open3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
    bb_corners = np.asarray(bb.get_box_points())

    maxs = np.max(bb_corners, axis = 0)
    mins = np.min(bb_corners, axis = 0)
    dimensions = maxs - mins
    sorted_dims = np.sort(dimensions)

    return sorted_dims[0], sorted_dims[1], sorted_dims[2]


def get_diag(corners, centre):
    """calculates max diagonal across object - important for sampling uniformly across crystal shift off centre

    Args:
        corners (list/ndarray): list of corner coordinates
        centre (list/ndarray): centre coordinate

    Returns:
        diag (float): calculated diagonal 
    """
    points = corners - centre
    distances = np.linalg.norm(points, axis = 1)
    return 2*np.max(distances)

class WulffCrystal(Custom):
    """
    Class object for Crystal based on wulff reconstruction

    By default use convex hull as way to make connections, otherwise can use n nearest neighbours approach.
    """

    def __init__(self, particle):

        corns = particle.wulff_convex.points
        corns = np.asarray(corns)

        corners_1 = corns
        self.corners = np.unique(corners_1.round(decimals =8), axis = 0)
        self.centre = np.mean(self.corners.T, axis = 1)
        self.diag = get_diag(self.corners, self.centre)
        self.s, self.i, self.l = s_i_l_from_wulff(particle)
        self.axis_align_morph = axis_align_s_i_l(particle)

        import itertools
        connections, faces = get_connections(self.corners)
        faces = sort_ascend(faces)
        n_faces = []
        for item in faces:
            temp = []
            for it in itertools.combinations(item, 2):
                temp.append(list(it))
            n_faces.append(temp)
        self.connections = sort_ascend(connections)
        self.faces = np.array(n_faces)

        self.rotated_corners = self.corners - 0.5*self.centre


def create_WulffCryst_fromSmorf(file):
    """ Create wulff construct object using above class from file downloaded
    from smorf.nl site. Takes lattice and surface energy inputs to generate wulff
    construction and set up slicable object.

    Args:
        file (string): JSON file to load

    Raises:
        ValueError: Need to make sure face distance interpretation is valid - should not be a problem
                    if file comes direct from smorf.

    Returns:
        wulff (crystal object): Crystal object to slice and work with in this model.
    """
    f = open(file)
    crystal_file = json.load(f)

    cell = crystal_file["cell"]
    forms = crystal_file["forms"]
    lattice = Lattice.from_parameters(cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"])
    if crystal_file["dconversion"] == "cartesian":
        surface_energies = {(forms[i]["h"], forms[i]["k"], forms[i]["l"]): forms[i]["d"] for i in range(len(forms))}
    elif crystal_file["dconversion"] == "none":#crystallographic
        surface_energies = {(forms[i]["h"], forms[i]["k"], forms[i]["l"]): forms[i]["d"]/np.sqrt(forms[i]["h"]**2+forms[i]["k"]**2+forms[i]["l"]**2) for i in range(len(forms))}
    else:
        raise ValueError("Smorf face distance interpretation not valid.")
    w = WulffShape(lattice, surface_energies.keys(), surface_energies.values())
    wulff = WulffCrystal(w)
    return wulff
