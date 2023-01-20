#newest iteration with properly sampling a spatially spherical space.
import numpy as np
import matplotlib.pyplot as plt

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

def nearest_neighbours(points, n):
    from sklearn.neighbors import KDTree

    tree = KDTree(points)
    d, i = tree.query(points, n+1)
    indices = i[:,1:]
    return indices

def get_connects(corners, n):
    connects = []
    ind = nearest_neighbours(corners, n)
    for i in range(len(corners)):
        for item in ind[i]:
            connects.append([i, item])

    return connects

def get_diag(corners, centre):
    points = corners - centre
    distances = np.linalg.norm(points, axis = 1)
    return 2*np.max(distances)

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


class Cuboid:
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
        self.connections = get_connections(self.corners)
        self.centre = np.mean(self.corners.T, axis = 1)
        self.diag = get_diag(self.corners, self.centre)

    def rotated_plot(self):
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

    def plot_intersect(self):
        fig = plt.figure()
        ax = plt.axes(projection = "3d")
        fig.axes.append(ax)
        
        nums = np.random.rand(7)
        shift = nums[0]
        axis = [nums[3], nums[4], nums[5]]
        angle = 360*nums[6]
        self.transform(shift, axis, angle, True)
        intersects, vertices = self.convex_hull()
        corners = self.rotated_corners
        ax.plot(corners[:,0],corners[:,1],corners[:,2],'k.')
        for item in self.connections:
            ax.plot([corners[item[0]][0], corners[item[1]][0]],[corners[item[0]][1], corners[item[1]][1]],[corners[item[0]][2], corners[item[1]][2]], 'k-')

        xx, yy = np.meshgrid(range(3), range(3))
        xx = xx -  1
        yy = yy - 1
        zz = yy*0
        ax.plot_surface(xx, yy, zz, alpha = 0.25)
        ax.plot(intersects[:,0], intersects[:,1], [0]*len(intersects[:,0]), 'r.')
        return fig, ax

    def plot(self, restrict = True):
        fig = plt.figure()
        ax = plt.axes(projection = "3d")
        fig.axes.append(ax)
        if restrict == True:
            ax.set_xlim([-0.5,0.5])
            ax.set_ylim([-0.5,0.5])
        else:
            pass
        self.plot_corners = self.corners - self.centre
        ax.plot(self.plot_corners[:,0],self.plot_corners[:,1],self.plot_corners[:,2],'k.')
        for item in self.connections:
            ax.plot([self.plot_corners[item[0]][0], self.plot_corners[item[1]][0]],[self.plot_corners[item[0]][1], 
            self.plot_corners[item[1]][1]],[self.plot_corners[item[0]][2], self.plot_corners[item[1]][2]], 'k-')
        return fig, ax

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

        return np.asarray(intersects)

    #then use convex hull of all points to find the shape of slice - this way it's guaranteed to be the correct shape
    #convex hull can be found using scipy's convex hull finder
    def convex_hull(self):
        """Use Convec Hull as a way to make the points into a polygon - essentially orders them correctly.

        Returns:
            intersects (ndarrray): points of intersections
            vertices (ndarray): order of intersection points needed to create the correct polygon enclosing
                                all points.
        """
        from scipy.spatial import ConvexHull
        intersects = self.xy_intersect()
        cHull = ConvexHull(intersects)
        vertices = cHull.vertices
        vertices = np.append(vertices, vertices[0])
        return intersects, vertices

    def create_img(self,points, hull, plot = False, multiplier =1000, man_mins = False, man_mins_val = 0):
        """Create binary img from convex hull.

        Args:
            points (ndarray): Points of edge intersection with plane.
            hull (ndarray): Convex hull order of points to make correct polygon.
            plot (bool, optional): Boolean switch to enable plotting. Defaults to False.

        Returns:
            binary img (ndarray): Binary image of intersection.
        """
        from matplotlib.path import Path

        if man_mins == False:
            mins = points[hull].min(axis = 0)
        else:
            mins = man_mins_val
        
        polygon=(points[hull] + np.asarray([0.1,0.1]) - mins)*multiplier
        poly_path=Path(polygon)
        maxs = polygon.max(axis = 0).astype("int64") + np.asarray([100,100])
        x, y = np.mgrid[:maxs[0], :maxs[1]]
        coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))
        mask = poly_path.contains_points(coors)
        img = mask.reshape(maxs[0], maxs[1])
        if plot == True:
            plt.imshow(mask.reshape(maxs[0], maxs[1]))
        else:
            pass
        return img
        
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

    def transform(self,plane_shift, axis, theta, degrees = True, manual_shift = False):
        """Performs transformation to the cuboid as passed to it - it will shit and rotate the cuboid as
        given by the parameters passed to it.

        Args:
            shift (ndarray): Move cuboid centre by the given vector.
            axis (ndarray: Axis of rotation.
            theta (float): Angle of rotation.
            degrees (bool, optional): Tell the function whether theta passed is in degrees or radians. Defaults to True - degrees.

        Returns:
            None.
        """
        
        #check for right angular unit
        if degrees == True:
            theta = (theta * np.pi)/180
        else:
            pass

        from scipy.spatial.transform import Rotation as R
        shifted_corners = self.corners - self.centre
        #rotationMatrix = rotation_matrix(axis, theta)
        rotationMatrix = R.random(1).as_matrix()[0]
        self.rotated_corners = np.around(np.asarray([np.dot(rotationMatrix, coord) for coord in shifted_corners]), decimals = 10)

        if manual_shift == False:
            shift = plane_shift*self.diag - 0.5*self.diag
            #shift = plane_shift*2*self.s - self.s
            #print(shift, plane_shift)
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
        nums = np.random.rand(3)
        #get spherical coordinates
        shift = nums[0] # r in spherical coordinates
        phi = np.arccos(2*nums[1]-1)
        theta = nums[2]*2*np.pi
        #convert spherical coordinates to axis (angles only)
        axis = np.asarray([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        # calculate angle of rotation:
        #z_axis = np.asarray([0,0,1])
        #rot_angle = np.arccos(np.dot(axis, z_axis))
        #rot_axis = np.cross(axis, z_axis)
        angle = np.random.rand(1)*2*np.pi
        v = (angle, axis, shift)
        return v
        

    def sample_slice(self, n_samples):
        """Second function that automates random slicing of cuboid object - different approach written 1008

        Args:
            n_samples (int): Number of random samples to be taken

        Returns:
            Res (list): Lengths and widths of random cuts through the cuboid.
        """
        #set seed
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
                intersects, vertices = self.convex_hull()
                mult = self.calc_multiplier(intersects)
                img = self.create_img(intersects, vertices, multiplier = mult)
                measurements = self.measure_props(img)
                if measurements == (0,0):
                    continue
                else:
                    res.append(measurements)
                    i += 1
        print(time.time() - start)
        return res

    #######################################################################################
    ##################      Functions to produce slice images       #######################
    #######################################################################################


    def create_show_img(self,points, hull, plot = False, multiplier =1000, man_mins = False, man_mins_val = 0):
        """Create binary img from convex hull.

        Args:
            points (ndarray): Points of edge intersection with plane.
            hull (ndarray): Convex hull order of points to make correct polygon.
            plot (bool, optional): Boolean switch to enable plotting. Defaults to False.

        Returns:
            binary img (ndarray): Binary image of intersection.
        """
        from matplotlib.path import Path

        from matplotlib.path import Path

        if man_mins == False:
            mins = points[hull].min(axis = 0)
        else:
            mins = man_mins_val
            
        polygon=(points[hull] + np.asarray([0.01,0.01]) - mins)*multiplier
        poly_path=Path(polygon)
        maxs = polygon.max(axis = 0).astype("int64") + np.asarray([1,1])
        x, y = np.mgrid[:maxs[0], :maxs[1]]
        coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))
        mask = poly_path.contains_points(coors)
        img = mask.reshape(maxs[0], maxs[1])
        if plot == True:
            plt.imshow(mask.reshape(maxs[0], maxs[1]))
        else:
            pass
        return img

    def random_img(self, auto_mult = True, multiplier = 100):

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
            axis = np.asarray([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            
            within = self.transform(shift, axis, angle, True)
            
        intersects, vertices = self.convex_hull()
        if auto_mult == True:
            mult = self.calc_multiplier(intersects, 100, False)
        else:
            mult = multiplier
        img = self.create_show_img(intersects, vertices, multiplier = multiplier)
        return img

    def create_10x10_slices(self, auto_mult = True, multiplier = 100):
        """_summary_

        Args:
            auto_mult (bool, optional): _description_. Defaults to True.
            multiplier (int, optional): _description_. Defaults to 100.

        Returns:
            _type_: _description_
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

