from .Cuboid import Cuboid
import numpy as np
import matplotlib.pyplot as plt

def get_diag(corners, centre):
    points = corners - centre
    distances = np.linalg.norm(points, axis = 1)
    return 2*np.max(distances)

def gen_ellipsoid(a,b,c,n):
    """Generate mesh of ellipsoid with n*n number of points.

    Args:
        a (_type_): _description_
        b (_type_): _description_
        c (_type_): _description_
        n (_type_): _description_
    """

    u = np.linspace(0,2*np.pi, n)
    v = u/2
    points = []
    for item in u:
        for item2 in v:
            points.append([a*np.cos(item)*np.sin(item2), b*np.sin(item)*np.sin(item2), c*np.cos(item2)])

    points = np.asarray(points)
    return points

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


class Ellipsoid(Cuboid):
    def __init__(self, s_over_i, i_over_l,size = 1, max_sizes = [1,1,1], n_points = 20):
        super().__init__(s_over_i, i_over_l, size, max_sizes)

        self.l = 1*size
        self.i = self.l*i_over_l
        self.s = self.i*s_over_i

        self.corners = gen_ellipsoid(self.s, self.i, self.l, n_points)
        self.connections = get_connections(self.corners)
        self.centre = np.mean(self.corners.T, axis = 1)
        self.diag = get_diag(self.corners, self.centre)