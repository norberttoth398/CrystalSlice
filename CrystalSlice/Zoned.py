import numpy as np
import matplotlib.pyplot as plt
from .Cuboid import Cuboid

class ZonedCuboid:
    """
    Zoned crystal modelling - interested in how slices will look like
    for zoned crystals.
    """
    def __init__(self, morph_list, zoning):
        """
        Initialise basic paremeters.

        morph_list - list of both S/I and I/L values for all the zones. Should be in order of going
                        outside zones in.
        zoning - list of zone sizes, first element should always be 1 and elements should decrease 
                        along list.
        """
        self.crysts = []
        #create cuboids for each zone
        for i in range(len(morph_list)):
            if i == 0:
                self.crysts.append(Cuboid(morph_list[i][0], morph_list[i][1], zoning[i]))
            else:
                self.crysts.append(Cuboid(morph_list[i][0], morph_list[i][1], zoning[i], max_sizes=[self.crysts[i-1].s,self.crysts[i-1].i,self.crysts[i-1].l]))
        
    def sample_variables(self):
        """sample random variables uniformly

        Returns:
            results (tuple): angle and axis of rotation as well as shift magnitude
        """
        nums = np.random.rand(3)
        #get spherical coordinates
        shift = nums[0] # r in spherical coordinates
        #convert spherical coordinates to axis (angles only)
        from scipy.spatial.transform import Rotation as R
        matrix = R.random(1).as_matrix()[0]
        # calculate angle of rotation:
        v = (matrix, shift)
        return v
    
    def sample_slice(self, auto_mult = True, multiplier = 100):
        """Function to create random slices from zoned crystals. Important considerations to make
        are to ensure all crystals and cut positions are consistent across all zones so "mins" 
        and "shift" variables are consistent and based on the outer zone (largest crystal) only.

        Args:
            auto_mult (bool, optional): Choose between dynamic multiplier generation or manual
                                        value to use. Defaults to True.
            multiplier (int, optional): Manual multiplier value if chosen. Defaults to 100.

        Returns:
            Full_img: Final image.
        """
        np.random.seed()
        within = 0
        while(within == 0):
            nums = np.random.rand(4)
            shift = nums[0]*self.crysts[0].diag - 0.5*self.crysts[0].diag
            
            matrix, shift = self.sample_variables()
            #transform all crystals
            crysts_within = [item.transform(shift,matrix, True) for item in self.crysts]
            #if outer zone is intersected then accept the current transform
            if crysts_within[0] == 1:
                within = 1
            else:
                within = 0

        intersects = []
        vertices = []
        i = 0
        while(crysts_within[i] == 1):
            inter, vert = self.crysts[i].xy_intersect()
            intersects.append(inter)
            vertices.append(vert)
            i += 1
            if i == len(crysts_within):
                break
            else:
                pass
        
        if auto_mult == True:
            mult = self.crysts[0].calc_multiplier(intersects[0], 175, False)
        else:
            mult = multiplier

        mins = intersects[0].min(axis = 0)
        img_list = [self.crysts[i].create_img(intersects[i], vertices[i], False, mult, True, mins) for i in range(len(intersects))]

        final_img = np.zeros_like(img_list[0]).astype("int64")
        for i in range(len(img_list)):
            temp_img = img_list[i].astype("int64")
            s1, s2 = temp_img.shape
            final_img[:s1, :s2] += temp_img*(i+1)

        return final_img


    def create_10x10_slices(self, auto_mult = True, multiplier = 100, size = 200):
        """Create 100 random slices and show them in a single image as done so in the
        CSDCorrections and CSDSlice papers.

        Args:
            auto_mult (bool, optional): Choose between dynamic multiplier generation or manual
                                        value to use. Defaults to True.
            multiplier (int, optional): Manual multiplier value if chosen. Defaults to 100.
            size (int, optional): Size of each tile, not final image is 10x10 of size. Defaults to 200.

        Returns:
            full_img: Final image.
        """
        full_img = np.zeros((size*10,size*10)).astype("int64")

        for n in range(100):
            i = int(n/10)
            j = n%10

            img = self.sample_slice(auto_mult, multiplier)
            s1, s2 = img.shape
            start = [int(i*size + size/2 - int(s1/2)), int(j*size + size/2 -int(s2/2))]
            end = [start[0] + s1, start[1] + s2]

            full_img[start[0]:end[0], start[1]:end[1]] = img

        return full_img

