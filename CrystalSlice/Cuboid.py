#newest iteration with properly sampling a spatially spherical space.
import numpy as np
import matplotlib.pyplot as plt
from .base import Custom, sort_ascend, get_connections

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




class Cuboid(Custom):
    """
    Class for a Cuboid object which is used as a simple model of Plagioclase crystals in this case.
    """
    def __init__(self, s_over_i, i_over_l, size = 1, max_sizes = [1,1,1]):
        """Initialise parameters of the object.

        Args:
            s_over_i (float): S/I ratio of cuboid.
            i_over_l (flota): I/L ratio of cuboid.
        """
        #define cuboid lengths - note l is always 1
        self.l = 1*size
        self.i = self.l*i_over_l
        self.s = self.i*s_over_i

        check = np.asarray([self.s, self.i, self.l]) <= np.asarray(max_sizes)
        if check.all() == True:
            pass
        else:
            m = np.min(np.divide(np.asarray(max_sizes), np.asarray([self.s, self.i, self.l])))
            [self.s, self.i, self.l] = np.asarray([self.s, self.i, self.l])*m
        

        #position of corners of cuboid
        self.corners = np.asarray([[0,0,0], [self.s,0,0], [self.s,self.i,0], [0, self.i, 0], 
                                    [0,0,self.l], [self.s,0,self.l], [self.s,self.i,self.l], [0, self.i, self.l]])

        #define connections - the vertices of the object.
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

        self.centre = np.mean(self.corners.T, axis = 1)
        self.diag = get_diag(self.corners, self.centre)
        self.rotated_corners = self.corners - 0.5*self.centre

    