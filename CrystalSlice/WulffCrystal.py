import numpy as np
import matplotlib.pyplot as plt
from .Cuboid import Cuboid
import open3d


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
    pcd.points = open3d.utility.Vector3dVector(corners(crystal))
    bb = open3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    axis_align = bb.get_axis_aligned_bounding_box()

    bb_corners = np.asarray(axis_align.get_box_points())
    maxs = np.max(bb_corners, axis = 1)
    mins = np.min(bb_corners, axis = 1)
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


class WulffCrystal(Cuboid):
    """
    Class object for Crystal based on wulff reconstruction
    """

    def __init__(self, particle, s_over_i=1, i_over_l=1, size = 1, max_sizes = [1,1,1]):
        super().__init__(s_over_i, i_over_l, size, max_sizes)

        corns = get_corners(particle)
        corns = np.asarray(corns)

        #connects = []
        #for i in range(len(corns)):
        #    connects.append([[i, j] for j in range(len(corns))])
        #connects = np.asarray(connects)
        #connects = connects.reshape(len(corns)**2, 2)
        connects = np.asarray(get_connections(corns))

        self.corners = corns
        self.connections = connects
        self.s, self.i, self.l = s_i_l_from_wulff(particle)
