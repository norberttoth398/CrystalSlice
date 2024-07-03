import numpy as np
import matplotlib.pyplot as plt
import open3d
from scipy.spatial.transform import Rotation as R

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

def s_i_l_BBOX(corner_points):
    """calculate bounding box S, I and L to relate to cuboid

    Args:
        corner_points (list/ndarray): list of corner coordinates

    Returns:
        S, I, L (float): dimensions of bounding box
    """
    
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
    """calculate axis aligned bounding box S, I and L to relate to cuboid

    Args:
        corner_points (list/ndarray): list of corner coordinates

    Returns:
        S, I, L (float): dimensions of bounding box
    """
    
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(corner_points)
    bb = open3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd.points)
    bb_corners = bb.get_box_points()

    maxs = np.max(bb_corners, axis = 0)
    mins = np.min(bb_corners, axis = 0)
    dimensions = maxs - mins
    sorted_dims = np.sort(dimensions)

    return sorted_dims[0], sorted_dims[1], sorted_dims[2]

def sort_ascend(item):
    """sort list in ascending order

    Args:
        item (list/ndarray): items to be sorted

    Returns:
        sorted (list/ndarray): sorted list of items
    """
    item = np.asarray(item)
    ind = np.argsort(item, axis = 1)
    new_item = []
    for i in range(len(ind)):
        new_item.append(item[i][ind[i]])

    return np.asarray(new_item)

def get_connections(points):
    """ method to calculate connections between corner points assuming
    convex shape

    Args:
        points (list/ndarray): corners

    Returns:
        connections (list/ndarray): resulting connections list
    """
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    connections = []
    faces = []
    for simplex in hull.simplices:
        for i in range(len(simplex)):
            first = i
            last = i+1
            if last >= len(simplex):
                last = 0
            else:
                pass
            connections.append([simplex[first], simplex[last]])
    return np.unique(connections, axis = 0), np.unique(hull.simplices, axis = 0)


######################################################################
############# OBJECT #################################################
######################################################################

class Custom:
    def __init__(self, corners, connections = None, faces = None, n = 3, s_over_i = 1, i_over_l = 1,size = 1, max_sizes = [1,1,1], n_points = 20):
        #super().__init__(corners)


        self.corners = np.multiply(np.asarray(corners), np.asarray(max_sizes))
        self.s, self.i, self.l = s_i_l_BBOX(self.corners)
        self.centre = np.mean(self.corners.T, axis = 1)
        self.diag = get_diag(self.corners, self.centre)
        self.axis_align_morph = axis_align_s_i_l(self.corners)
        if connections is not None:
            assert faces is not None
            self.connections = np.asarray(connections)
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
        else:
            #convex assumption
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

    def rotated_plot(self):
        """create a plot of rotated object

        """
        fig = plt.figure()
        ax = plt.axes(projection = "3d")
        fig.axes.append(ax)
        corners = self.rotated_corners
        ax.plot(corners[:,0],corners[:,1],corners[:,2],'k.')
        for item in self.connections:
            ax.plot([corners[item[0]][0], corners[item[1]][0]],[corners[item[0]][1], corners[item[1]][1]],[corners[item[0]][2], corners[item[1]][2]], 'k-')

        xx, yy = np.meshgrid(range(3), range(3))
        xx = xx -  1
        yy = yy - 1
        zz = yy*0
        ax.plot_surface(xx, yy, zz, alpha = 0.25)
        return fig, ax

    
    def plot(self, restrict = True, fig = None, ax = None):
        """plot object in 3D

        Args:
            restrict (bool, optional): Restrict axes labels or not?. Defaults to True.
            fig (_type_, optional): matplotlib figure object for plotting. Defaults to None.
            ax (_type_, optional): matplotlib axis object to plot onto. Defaults to None.

        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        if fig is None:
            fig = plt.figure()
            ax = plt.axes(projection = "3d")
            fig.axes.append(ax)

        self.plot_corners = self.corners - self.centre
        if restrict == True:
            ax.set_xlim([-0.5,0.5])
            ax.set_ylim([-0.5,0.5])
        else:
            ax.set_xlim([np.min(self.plot_corners)*1.1,np.max(self.plot_corners)*1.1])
            ax.set_ylim([np.min(self.plot_corners)*1.1,np.max(self.plot_corners)*1.1])
            ax.set_zlim([np.min(self.plot_corners)*1.1,np.max(self.plot_corners)*1.1])
        
        for item in self.faces:
            ps = np.unique(item)
            ax.add_collection(Poly3DCollection([self.plot_corners[ps]],facecolors='w', linewidths=0, alpha=0.5))
        ax.plot(self.plot_corners[:,0],self.plot_corners[:,1],self.plot_corners[:,2],'k.')
        for item in self.connections:
            ax.plot([self.plot_corners[item[0]][0], self.plot_corners[item[1]][0]],[self.plot_corners[item[0]][1], 
            self.plot_corners[item[1]][1]],[self.plot_corners[item[0]][2], self.plot_corners[item[1]][2]], 'k-')
        return fig, ax

    
        
    def measure_props(self, img):
        """Measure the properties of the binary img - ie the slice/cut of the cuboid taken.

        Args:
            img (ndarray): binary img of the slice.

        Returns:
            length, width: length and width of slice taken.
        """
        from skimage import measure

        label = measure.label(img.astype("int64"))
        props = measure.regionprops(label)
        #if len props == 0 return [0,0] and have function that makes use of that info to restart loop
        if len(props) == 0:
            return (0,0)
        else:
            l = props[0].major_axis_length
            w = props[0].minor_axis_length

            return (l,w)

    def transform(self,plane_shift, matrix, manual_shift = False):
        """Performs transformation to the cuboid as passed to it - it will shift and rotate the cuboid as
        given by the parameters passed to it.

        Args:
            shift (ndarray): Move cuboid centre by the given vector.
            axis (ndarray: Axis of rotation.
            theta (float): Angle of rotation.
            degrees (bool, optional): Tell the function whether theta passed is in degrees or radians. Defaults to True - degrees.

        Returns:
            None.
        """
        

        
        shifted_corners = self.corners - self.centre
        self.rotated_corners = np.around(np.asarray([np.dot(matrix, coord) for coord in shifted_corners]), decimals = 10)

        if manual_shift == False:
            shift = plane_shift*self.diag - 0.5*self.diag
        else:
            shift = plane_shift
        
        self.rotated_corners[:,2] = self.rotated_corners[:,2] + shift
        z_max = np.max(self.rotated_corners[:,2])
        z_min = np.min(self.rotated_corners[:,2])
        if z_min > 0:
            return 0
        elif z_max < 0:
            return 0
        else:
            return 1

    
    def calc_multiplier(self,intersects, target = 100, max = True):
        """
        Function to dynamically calculate the multiplier required for create_img
        function
        """
        inter_min = np.min(intersects,0)
        inter_max = np.max(intersects,0)
        diff = inter_max[:2] - inter_min[:2]
        if max == True:
            multiplier = np.max(target/diff)
        else:
            multiplier = np.min(target/diff)
        if multiplier > 10000:
            return 10000
        else:
            return multiplier

    def sample_variables(self):
        """sample random variables uniformly

        Returns:
            results (tuple): angle and axis of rotation as well as shift magnitude
        """
        nums = np.random.rand(3)
        #get spherical coordinates
        shift = nums[0] # r in spherical coordinates
        #convert spherical coordinates to axis (angles only)
        
        matrix = R.random(1).as_matrix()[0]
        # calculate angle of rotation:
        v = (matrix, shift)
        return v


    def random_img(self, auto_mult = True, multiplier = 100):
        """Generate random image, used for generating 10x10
        figure
        """

        #set seed
        np.random.seed()
        within = 0
        while(within == 0):
            nums = np.random.rand(4)
            shift = nums[0]
            angle = 360*nums[1]
            
            #get axis in spherical coordinates
            theta = nums[2]*np.pi
            phi = nums[3]*2*np.pi
            #convert to spherical coordinates
            matrix = R.random(1).as_matrix()[0]
            
            within = self.transform(shift, matrix)
            
        intersects, vertices = self.xy_intersect()
        if auto_mult == True:
            mult = self.calc_multiplier(intersects, 100, False)
        else:
            mult = multiplier
        img = self.create_img(intersects, vertices, multiplier = multiplier)
        return img

    def create_10x10_slices(self, auto_mult = True, multiplier = 100):
        """create array image of random slcies
        """
        full_img = np.zeros((2000,2000)).astype("bool")

        for n in range(100):
            i = int(n/10)
            j = n%10

            img = self.random_img(auto_mult, multiplier)
            s1, s2 = img.shape
            start = [i*200 + 100 - int(s1/2), j*200 + 100 -int(s2/2)]
            end = [start[0] + s1, start[1] + s2]

            full_img[start[0]:end[0], start[1]:end[1]] = img

        return full_img




    def xy_intersect(self, rotated = True):
        """Produce an intersect of the object about the xy axis.

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
        face_dict = {}
        #need to check that xy axis lies inbetween corner pairs - if so then it intercepts
        for i in range(len(self.connections)):
            ind = self.connections[i]
            if corners[ind[0]][2] > 0 and corners[ind[1]][2] > 0:
                pass
            elif corners[ind[0]][2] < 0 and corners[ind[1]][2] < 0:
                pass
            else:
                cut_ind.append(ind)
                face_dict[i] = np.where((self.faces == ind).all(-1) == True)[0]
        #print(face_dict)
        intersects = []
        for item in cut_ind:
            #vector between two corners is simply a - b, find how far along it is, z = 0 and point of intersection is found.
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
        #print(cut_ind)
        new_cut_inds = []
        for c in cut_ind:
            item = self.connections.tolist().index(c.tolist())
            new_cut_inds.append(item)
        #print(new_cut_inds)

        return np.array(intersects), face_dict



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
        #print(slice_connects)
        for item in np.unique(np.array(list(slice_connects.values())).ravel()):
            inds = np.where(np.array(list(slice_connects.values())) == item)[0]

            for n in range(len(inds)-1):          
                start = scaled_points[inds[n]]
                stop = scaled_points[inds[n+1]]
                #print(start, stop)

                rr, cc = draw.line(start[0], start[1], stop[0], stop[1])

                mask[rr, cc] = 1*img_val
        
        import scipy.ndimage as ndimage    
        mask_n = ndimage.binary_fill_holes(mask == img_val)
        mask[mask_n] = img_val
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
            matrix, shift = self.sample_variables()
            within = self.transform(shift,matrix)
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
        

    def plot_intersect(self):
            """ create plot showing intersecting plane on randomly oriented object

            Returns:
                _type_: _description_
            """
            fig = plt.figure()
            ax = plt.axes(projection = "3d")
            fig.axes.append(ax)
            
            nums = np.random.rand(7)
            shift = nums[0]
            matrix = R.random(1).as_matrix()[0]
            self.transform(shift, matrix)
            intersects, vertices = self.xy_intersect()
            #print(intersects)
            #print(vertices)
            corners = self.rotated_corners
            ax.plot(corners[:,0],corners[:,1],corners[:,2],'k.')
            for item in self.connections:
                ax.plot([corners[item[0]][0], corners[item[1]][0]],[corners[item[0]][1], corners[item[1]][1]],[corners[item[0]][2], corners[item[1]][2]], 'k-')

            xx, yy = np.meshgrid(range(3), range(3))
            xx = xx -  1
            yy = yy - 1
            zz = yy*0
            ax.plot_surface(xx, yy, zz, alpha = 0.25)
            try:
                ax.plot(intersects[:,0], intersects[:,1], [0]*len(intersects[:,0]), 'r.')
                return fig, ax
            except:
                self.plot_intersect()
            
    
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