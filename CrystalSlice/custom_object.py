import numpy as np
import matplotlib.pyplot as plt
from .Cuboid import Cuboid
import open3d

def get_diag(corners, centre):
    points = corners - centre
    distances = np.linalg.norm(points, axis = 1)
    return 2*np.max(distances)

def s_i_l_BBOX(corner_points):
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(corner_points)
    bb = open3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    bb_corners = np.dot(bb.get_box_points(), bb.R)

    maxs = np.max(bb_corners, axis = 0)
    mins = np.min(bb_corners, axis = 0)
    dimensions = maxs - mins
    sorted_dims = np.sort(dimensions)

    return sorted_dims[0], sorted_dims[1], sorted_dims[2]

def axis_align_s_i_l(corner_points):
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(corner_points)
    bb = open3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
    bb_corners = bb.get_box_points()

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

######################################################################
############# OBJECT #################################################
######################################################################

class Custom(Cuboid):
    def __init__(self, corners, connections = None, convex = True, n = 3, s_over_i = 1, i_over_l = 1,size = 1, max_sizes = [1,1,1], n_points = 20):
        super().__init__(s_over_i, i_over_l, size, max_sizes)


        self.corners = np.asarray(corners)
        self.s, self.i, self.l = s_i_l_BBOX(self.corners)
        self.centre = np.mean(self.corners.T, axis = 1)
        self.diag = get_diag(self.corners, self.centre)
        self.axis_align_morph = axis_align_s_i_l(self.corners)
        if connections is not None:
            self.connections = np.asarray(connections)
        else:
            if convex is True:
                self.connections = get_connections(self.corners)
            elif convex is False:
                self.connections = get_connects(self.corners, n)
            else:
                raise ValueError("Convex input variable must be boolean (True or False).")

        