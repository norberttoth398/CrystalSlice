import numpy as np
import matplotlib.pyplot as plt
from .Cuboid import Cuboid
import open3d

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    from https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    import math
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

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

def sort_ascend(item):
    item = np.asarray(item)
    ind = np.argsort(item, axis = 1)
    new_item = []
    for i in range(len(ind)):
        new_item.append(item[i][ind[i]])

    return np.asarray(new_item)

######################################################################
############# OBJECT #################################################
######################################################################

class Custom(Cuboid):
    def __init__(self, corners, connections, faces, convex = True, n = 3, s_over_i = 1, i_over_l = 1,size = 1, max_sizes = [1,1,1], n_points = 20):
        super().__init__(s_over_i, i_over_l, size, max_sizes)


        self.corners = np.multiply(np.asarray(corners), np.asarray(max_sizes))
        self.s, self.i, self.l = s_i_l_BBOX(self.corners)
        self.centre = np.mean(self.corners.T, axis = 1)
        self.diag = get_diag(self.corners, self.centre)
        self.axis_align_morph = axis_align_s_i_l(self.corners)
        #if connections is not None:
        self.connections = np.asarray(connections)
        #else:
        #    if convex is True:
        #        self.connections = get_connections(self.corners)
        #    elif convex is False:
        #        self.connections = get_connects(self.corners, n)
        #    else:
        #        raise ValueError("Convex input variable must be boolean (True or False).")

        self.connections = sort_ascend(self.connections)
        new_faces = []
        for f in faces:
            new_faces.append(sort_ascend(f))
        face_inds = []
        for f in new_faces:
            item = []
            for f_element in f:
                item.append(self.connections.tolist().index(f_element.tolist()))
            face_inds.append(item)
        self.faces = np.asarray(face_inds)
        self.rotated_corners = self.corners - 0.5*self.centre


####################################################################################
############## Re-writing some parts of the slicing mechanism ######################
####################################################################################

    def xy_intersect(self, rotated = True):
        """Produce an intersect of the cuboid about the xy axis.

        Args:
            rotated (bool, optional): Tell function whether to use rotated object or not; if True then the object
                                        must have been rotated first. Defaults to True.

        Returns:
            intersects (ndarray): Points of intersection.
        """
        if rotated == True:
            corners = self.rotated_corners
        else:
            corners = self.corners

        cut_ind = []
        #need to check that xy axis lies inbetween corner pairs - if so then it intercepts
        for i in range(len(self.connections)):
            ind = self.connections[i]
            if corners[ind[0]][2] > 0 and corners[ind[1]][2] > 0:
                pass
            elif corners[ind[0]][2] < 0 and corners[ind[1]][2] < 0:
                pass
            else:
                cut_ind.append(ind)
        
        intersects = []
        for item in cut_ind:
            #vector between two corners is simply a - b, find how far along it, z = 0 and point of intersection is found.
            corner_1 = corners[item[0]]
            corner_2 = corners[item[1]]

            vector = corner_2 - corner_1
            if vector[2] == 0:
                intersects.append([corner_1[0], corner_1[1]])
                intersects.append([corner_2[0], corner_2[1]])
            else:
                prop_along_v = np.abs(corner_1[2]/vector[2])
                inter = corner_1 + prop_along_v*vector
                # x and y values that we need are now just inter[0] and inter[1]
                intersects.append([inter[0], inter[1]])

        new_cut_inds = []
        for c in cut_ind:
            item = self.connections.tolist().index(c.tolist())
            new_cut_inds.append(item)

        face_cut_ind = []
        for ind in new_cut_inds:
            af = np.zeros_like(self.faces)
            af[self.faces == ind] = 1
            face_s = np.sum(af, axis = 1)
            face_cut_ind.append(face_s)
        t = np.where(np.asarray(face_cut_ind).T == 1)
        u = np.unique(t[1])
        relate = np.asarray([np.asarray(t[0])[t[1] == un] for un in u])
        slice_connect = []
        
        for el in np.unique(relate):
            double_val = np.asarray(np.where(relate.ravel() == el))
            double_val = (double_val/2).astype("int64")
            slice_connect.append(double_val[0].tolist())

        import itertools
        final_connects = []
        for i in range(len(slice_connect)):
            if len(slice_connect[i]) == 2:
                final_connects.append(slice_connect[i])
            elif len(slice_connect[i]) == 1:
                pass#shouldn't be the case
            else:
                final_connects.append(list(itertools.combinations(slice_connect[i],2)))

        return np.asarray(intersects), np.asarray(final_connects)


    def create_img(self,points, slice_connects, plot = False, multiplier =1000, man_mins = False, man_mins_val = 0, img_val = 1):
        """Create binary img from convex hull.

        Args:
            points (ndarray): Points of edge intersection with plane.
            hull (ndarray): Convex hull order of points to make correct polygon.
            plot (bool, optional): Boolean switch to enable plotting. Defaults to False.

        Returns:
            binary img (ndarray): Binary image of intersection.
        """
        from matplotlib.path import Path
        from skimage import draw

        if man_mins == False:
            mins = points.min(axis = 0)
        else:
            mins = man_mins_val
        
        scaled_points=((points + np.asarray([int(10/multiplier),int(10/multiplier)]) - mins)*multiplier).astype("int64")
        maxs = scaled_points.max(axis = 0).astype("int64") + np.asarray([1,1])

        mask = np.zeros(maxs)
        for item in scaled_points:
            mask[item[0], item[1]] = 1*img_val
        for inds in slice_connects:          
            start = scaled_points[inds[0]]
            stop = scaled_points[inds[1]]
            #print(start, stop)

            rr, cc = draw.line(start[0], start[1], stop[0], stop[1])

            mask[rr, cc] = 1*img_val
        if plot == True:
            plt.imshow(mask.reshape(maxs[0], maxs[1]))
        else:
            pass
        return mask    

    def sample_slice(self, n_samples, return_img = False, return_res = True):
        """Second function that automates random slicing of cuboid object - different approach written 1008

        Args:
            n_samples (int): Number of random samples to be taken

        Returns:
            Res (list): Lengths and widths of random cuts through the cuboid.
        """
        #set random seed
        np.random.seed()
        import time
        start = time.time()

        res = []
        i = 0
        while(len(res) < n_samples):
            if i%100 == 0:
                print(i)
            else:
                pass
            outside = True
            #need to generate random numbers for rotation axis (3), shift (3), angle (1)
            angle, axis, shift = self.sample_variables()
            within = self.transform(shift, axis, angle, False)
            if within == 0:
                continue
            else:
                intersects, vertices = self.xy_intersect()
                #print(intersects, vertices)
                mult = self.calc_multiplier(intersects, target=250)
                img = self.create_img(intersects, vertices, multiplier = mult)
                measurements = self.measure_props(img)
                if measurements == (0,0):
                    continue
                else:
                    res.append(measurements)
                    i += 1
        print(time.time() - start)
        if return_img is True:
            return img
        elif return_res is True:
            return res
        else:
            return None
