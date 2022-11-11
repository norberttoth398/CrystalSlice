import numpy as np
import matplotlib.pyplot as plt
from .Cuboid import Cuboid
import open3d
import json
from ase.spacegroup import crystal
from wulffpack import SingleCrystal

def nearest_neighbours(points, n):
    from sklearn.neighbors import KDTree

    tree = KDTree(points)
    d, i = tree.query(points, n)
    indices = i[:,1:]
    dist = d[:,1:]
    return indices, dist

def get_connects(corners, n):
    connects = []
    ind, dist = nearest_neighbours(corners, len(corners))
    for i in range(len(corners)):
        k = 0
        for item in np.unique(dist[i]):
            if k >= n:
                pass
            else:
                d = dist[i]
                k += len(d[d == item])
        for j in range(k):
            it = ind[i][j]
            connects.append([i, it])

    return connects

def get_corners(particle):
    """Returns the corners (vertices) of the particle."""
    vertices = []
    for form in particle.forms:
        if form.parent_miller_indices == 'twin':
            continue
        for facet in form.facets:
            for vertex in facet.vertices:
                vertices.append(vertex)

    _, unique_ids = np.unique(vertices, axis = 0, return_index=True)
    vertices = np.asarray(vertices)
    unique_vertices = vertices[unique_ids]
    return unique_vertices

def s_i_l_from_wulff(crystal):
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(get_corners(crystal))
    bb = open3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    bb_corners = np.dot(bb.get_box_points(), bb.R.T)

    maxs = np.max(bb_corners, axis = 1)
    mins = np.min(bb_corners, axis = 1)
    dimensions = maxs - mins
    sorted_dims = np.sort(dimensions)

    return sorted_dims[0], sorted_dims[1], sorted_dims[2]

def axis_align_s_i_l(crystal):
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(get_corners(crystal))
    bb = open3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
    bb_corners = np.asarray(bb.get_box_points())

    maxs = np.max(bb_corners, axis = 0)
    mins = np.min(bb_corners, axis = 0)
    dimensions = maxs - mins
    sorted_dims = np.sort(dimensions)

    return sorted_dims[0], sorted_dims[1], sorted_dims[2]

def get_connections(points):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    connections = []
    for simplex in hull.simplices:
        for i in range(len(simplex)):
            first = i
            last = i+1
            if last >= len(simplex):
                last = 0
            else:
                pass
            connections.append([simplex[first], simplex[last]])
    return connections

def get_diag(corners, centre):
    points = corners - centre
    distances = np.linalg.norm(points, axis = 1)
    return 2*np.max(distances)

class WulffCrystal(Cuboid):
    """
    Class object for Crystal based on wulff reconstruction

    By default use convex hull as way to make connections, otherwise can use n nearest neighbours approach.
    """

    def __init__(self, particle, s_over_i=1, i_over_l=1, size = 1, max_sizes = [1,1,1], convex = True, n = 3):
        super().__init__(s_over_i, i_over_l, size, max_sizes)

        corns = get_corners(particle)
        corns = np.asarray(corns)

        #connects = []
        #for i in range(len(corns)):
        #    connects.append([[i, j] for j in range(len(corns))])
        #connects = np.asarray(connects)
        #connects = connects.reshape(len(corns)**2, 2)
        #connects = np.asarray(get_connections(corns))

        corners_1 = corns
        #self.connections = connects
        self.corners = np.unique(corners_1.round(decimals =8), axis = 0)
        self.centre = np.mean(self.corners.T, axis = 1)
        self.diag = get_diag(self.corners, self.centre)
        self.s, self.i, self.l = s_i_l_from_wulff(particle)
        self.axis_align_morph = axis_align_s_i_l(particle)
        if convex == True:
            self.connections = get_connections(self.corners)
        else:
            self.connections = get_connects(self.corners, n)


def create_WulffCryst_fromSmorf(file):
    f = open(file)
    crystal_file = json.load(f)

    cell = crystal_file["cell"]
    forms = crystal_file["forms"]
    sym = crystal('P', [(0,0,0)], cellpar=[cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"]])
    if crystal_file["dconversion"] == "cartesian":
        surface_energies = {(forms[i]["h"], forms[i]["k"], forms[i]["l"]): forms[i]["d"] for i in range(len(forms))}
    elif crystal_file["dconversion"] == "none":#crystallographic
        surface_energies = {(forms[i]["h"], forms[i]["k"], forms[i]["l"]): forms[i]["d"]/np.sqrt(forms[i]["h"]**2+forms[i]["k"]**2+forms[i]["l"]) for i in range(len(forms))}
    else:
        raise ValueError("Smorf face distance interpretation not valid.")
    particle = SingleCrystal(surface_energies, sym)
    wulffcryst = WulffCrystal(particle)
    return wulffcryst
