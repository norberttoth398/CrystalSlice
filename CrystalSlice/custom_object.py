import numpy as np
import matplotlib.pyplot as plt
from .Cuboid import Cuboid
import open3d

# def get_diag(corners, centre):
#     points = corners - centre
#     distances = np.linalg.norm(points, axis = 1)
#     return 2*np.max(distances)

# def s_i_l_BBOX(corner_points):
    
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(corner_points)
#     bb = open3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
#     bb_corners = np.dot(bb.get_box_points(), bb.R)

#     maxs = np.max(bb_corners, axis = 0)
#     mins = np.min(bb_corners, axis = 0)
#     dimensions = maxs - mins
#     sorted_dims = np.sort(dimensions)

#     return sorted_dims[0], sorted_dims[1], sorted_dims[2]

# def axis_align_s_i_l(corner_points):
    
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(corner_points)
#     bb = open3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
#     bb_corners = bb.get_box_points()

#     maxs = np.max(bb_corners, axis = 0)
#     mins = np.min(bb_corners, axis = 0)
#     dimensions = maxs - mins
#     sorted_dims = np.sort(dimensions)

#     return sorted_dims[0], sorted_dims[1], sorted_dims[2]

# def sort_ascend(item):
#     item = np.asarray(item)
#     ind = np.argsort(item, axis = 1)
#     new_item = []
#     for i in range(len(ind)):
#         new_item.append(item[i][ind[i]])

#     return np.asarray(new_item)

# def get_connections(points):
#     from scipy.spatial import ConvexHull
#     hull = ConvexHull(points)
#     connections = []
#     faces = []
#     for simplex in hull.simplices:
#         for i in range(len(simplex)):
#             first = i
#             last = i+1
#             if last >= len(simplex):
#                 last = 0
#             else:
#                 pass
#             connections.append([simplex[first], simplex[last]])
#     return np.unique(connections, axis = 0), np.unique(hull.simplices, axis = 0)

# def get_diag(corners, centre):
#     points = corners - centre
#     distances = np.linalg.norm(points, axis = 1)
#     return 2*np.max(distances)

# ######################################################################
# ############# OBJECT #################################################
# ######################################################################

# class Custom(Cuboid):
#     def __init__(self, corners, connections = None, faces = None, n = 3, s_over_i = 1, i_over_l = 1,size = 1, max_sizes = [1,1,1], n_points = 20):
#         super().__init__(s_over_i, i_over_l, size, max_sizes)


#         self.corners = np.multiply(np.asarray(corners), np.asarray(max_sizes))
#         self.s, self.i, self.l = s_i_l_BBOX(self.corners)
#         self.centre = np.mean(self.corners.T, axis = 1)
#         self.diag = get_diag(self.corners, self.centre)
#         self.axis_align_morph = axis_align_s_i_l(self.corners)
#         if connections is not None:
#             assert faces is not None
#             self.connections = np.asarray(connections)
#             new_faces = []
#             for f in faces:
#                 new_faces.append(sort_ascend(f))
#             face_inds = []
#             for f in new_faces:
#                 item = []
#                 for f_element in f:
#                     item.append(self.connections.tolist().index(f_element.tolist()))
#                 face_inds.append(item)
#             self.faces = np.asarray(face_inds)
#         else:
#             #convex assumption
#             import itertools
#             connections, faces = get_connections(self.corners)
#             faces = sort_ascend(faces)
#             n_faces = []
#             for item in faces:
#                 temp = []
#                 for it in itertools.combinations(item, 2):
#                     temp.append(list(it))
#                 n_faces.append(temp)
#             self.connections = sort_ascend(connections)
#             self.faces = np.array(n_faces)

        
        
#         self.rotated_corners = self.corners - 0.5*self.centre


# ####################################################################################
# ############## Re-writing some parts of the slicing mechanism ######################
# ####################################################################################

#     def xy_intersect(self, rotated = True):
#         """Produce an intersect of the object about the xy axis.

#         Args:
#             rotated (bool, optional): Tell function whether to use rotated object or not; if True then the object
#                                         must have been rotated first. Defaults to True.

#         Returns:
#             intersects (ndarray): Points of intersection.
#         """
#         if rotated == True:
#             corners = self.rotated_corners
#         else:
#             corners = self.corners

#         cut_ind = []
#         face_dict = {}
#         #need to check that xy axis lies inbetween corner pairs - if so then it intercepts
#         for i in range(len(self.connections)):
#             ind = self.connections[i]
#             if corners[ind[0]][2] > 0 and corners[ind[1]][2] > 0:
#                 pass
#             elif corners[ind[0]][2] < 0 and corners[ind[1]][2] < 0:
#                 pass
#             else:
#                 cut_ind.append(ind)
#                 face_dict[i] = np.where((self.faces == ind).all(-1) == True)[0]
#         print(face_dict)
#         intersects = []
#         for item in cut_ind:
#             #vector between two corners is simply a - b, find how far along it is, z = 0 and point of intersection is found.
#             corner_1 = corners[item[0]]
#             corner_2 = corners[item[1]]

#             vector = corner_2 - corner_1
#             if vector[2] == 0:
#                 intersects.append([corner_1[0], corner_1[1]])
#                 intersects.append([corner_2[0], corner_2[1]])
#             else:
#                 prop_along_v = np.abs(corner_1[2]/vector[2])
#                 inter = corner_1 + prop_along_v*vector
#                 # x and y values that we need are now just inter[0] and inter[1]
#                 intersects.append([inter[0], inter[1]])
#         #print(cut_ind)
#         new_cut_inds = []
#         for c in cut_ind:
#             item = self.connections.tolist().index(c.tolist())
#             new_cut_inds.append(item)
#         print(new_cut_inds)

#         return np.array(intersects), face_dict



#     def create_img(self,points, slice_connects, plot = False, multiplier =1000, man_mins = False, man_mins_val = 0, img_val = 1):
#         """Create binary img from convex hull.

#         Args:
#             points (ndarray): Points of edge intersection with plane.
#             hull (ndarray): Convex hull order of points to make correct polygon.
#             plot (bool, optional): Boolean switch to enable plotting. Defaults to False.

#         Returns:
#             binary img (ndarray): Binary image of intersection.
#         """
#         from matplotlib.path import Path
#         from skimage import draw

#         if man_mins == False:
#             mins = points.min(axis = 0)
#         else:
#             mins = man_mins_val
        
#         scaled_points=((points + np.asarray([int(10/multiplier),int(10/multiplier)]) - mins)*multiplier).astype("int64")
#         maxs = scaled_points.max(axis = 0).astype("int64") + np.asarray([1,1])

#         mask = np.zeros(maxs)
#         for item in scaled_points:
#             mask[item[0], item[1]] = 1*img_val
#         print(slice_connects)
#         for item in np.unique(np.array(list(slice_connects.values())).ravel()):
#             inds = np.where(np.array(list(slice_connects.values())) == item)[0]

#             for n in range(len(inds)-1):          
#                 start = scaled_points[inds[n]]
#                 stop = scaled_points[inds[n+1]]
#                 print(start, stop)

#                 rr, cc = draw.line(start[0], start[1], stop[0], stop[1])

#                 mask[rr, cc] = 1*img_val
        
#         import scipy.ndimage as ndimage    
#         mask_n = ndimage.binary_fill_holes(mask == img_val)
#         mask[mask_n] = img_val
#         if plot == True:
#             plt.imshow(mask.reshape(maxs[0], maxs[1]))
#         else:
#             pass
#         return mask    

#     def sample_slice(self, n_samples, return_img = False, return_res = True):
#         """Second function that automates random slicing of cuboid object - different approach written 1008

#         Args:
#             n_samples (int): Number of random samples to be taken

#         Returns:
#             Res (list): Lengths and widths of random cuts through the cuboid.
#         """
#         #set random seed
#         np.random.seed()
#         import time
#         start = time.time()

#         res = []
#         i = 0
#         while(len(res) < n_samples):
#             if i%100 == 0:
#                 print(i)
#             else:
#                 pass
#             outside = True
#             #need to generate random numbers for rotation axis (3), shift (3), angle (1)
#             angle, axis, shift = self.sample_variables()
#             within = self.transform(shift, axis, angle, False)
#             if within == 0:
#                 continue
#             else:
#                 intersects, vertices = self.xy_intersect()
#                 print(intersects, vertices)
#                 mult = self.calc_multiplier(intersects, target=250)
#                 img = self.create_img(intersects, vertices, multiplier = mult)
#                 measurements = self.measure_props(img)
#                 if measurements == (0,0):
#                     continue
#                 else:
#                     res.append(measurements)
#                     i += 1
#         print(time.time() - start)
#         if return_img is True:
#             return img
#         elif return_res is True:
#             return res
#         else:
#             return None
        

#     def plot_intersect(self):
#             fig = plt.figure()
#             ax = plt.axes(projection = "3d")
#             fig.axes.append(ax)
            
#             nums = np.random.rand(7)
#             shift = nums[0]
#             axis = [nums[3], nums[4], nums[5]]
#             angle = 360*nums[6]
#             self.transform(shift, axis, angle, True)
#             intersects, vertices = self.xy_intersect()
#             print(intersects)
#             print(vertices)
#             corners = self.rotated_corners
#             ax.plot(corners[:,0],corners[:,1],corners[:,2],'k.')
#             for item in self.connections:
#                 ax.plot([corners[item[0]][0], corners[item[1]][0]],[corners[item[0]][1], corners[item[1]][1]],[corners[item[0]][2], corners[item[1]][2]], 'k-')

#             xx, yy = np.meshgrid(range(3), range(3))
#             xx = xx -  1
#             yy = yy - 1
#             zz = yy*0
#             ax.plot_surface(xx, yy, zz, alpha = 0.25)
#             ax.plot(intersects[:,0], intersects[:,1], [0]*len(intersects[:,0]), 'r.')
#             return fig, ax