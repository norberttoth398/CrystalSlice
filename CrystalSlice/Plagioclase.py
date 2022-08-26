import numpy as np
import matplotlib.pyplot as plt
from Cuboid import Cuboid

class PlagCrystal(Cuboid):
    """
    Class object for Plag Crystal slicing
    """

    def __init__(self, s_over_i, i_over_l, size = 1, max_sizes = [1,1,1]):
        super().__init__(s_over_i, i_over_l, size, max_sizes)

        #corners and shape based off this image: https://www.researchgate.net/profile/Michael-Higgins-3/publication/31264562/figure/fig5/AS:650757183389706@1532164011563/Schematic-illustration-of-the-response-of-plagioclase-crystal-faces-to-the-chemical.png
        #All angles fixed at 45 degrees here for simplicity but can easily be changed

        s = self.s
        i = self.i
        l = self.l
        i_int = i - s
        l_int = l - i_int

        self.corners = np.asarray([[0,0.5*i,0], [s, 0.5*i, 0], [0,0.5*s, 0.5*i_int], [s, 0.5*s, 0.5*i_int], [0.5*s,0,0.5*i_int+0.5*s], [0.5*s,0,l-0.5*i_int-0.5*s], 
                        [0,0.5*s, l-0.5*i_int], [s, 0.5*s, l-0.5*i_int], [0,0.5*i,l], [s, 0.5*i, l], [0,i-0.5*s, l-0.5*i_int], [s, i-0.5*s, l-0.5*i_int],
                        [0.5*s,i,l-0.5*i_int-0.5*s], [0.5*s,i,0.5*i_int+0.5*s], [0,i - 0.5*s, 0.5*i_int], [s, i - 0.5*s, 0.5*i_int]])
        self.connections = np.asarray([[0,1], [0,2], [1,3], [2,4], [2,6], [3,4], [3,7],[4,5], [5,6], [5,7], [6,8],[7,9],[8,9], [8,10], [9,11], [10,12],[10,14],
                            [11,12], [11,15],[12,13], [13,14], [13,15], [14,0], [15,1]])

class ProportionalPlagCrystal(Cuboid):
    """
    Class object for Plag Crystal slicing
    """

    def __init__(self, s_over_i, i_over_l, size = 1, del_l = 0.1, del_i = 0.1, max_sizes = [1,1,1]):
        super().__init__(s_over_i, i_over_l, size, max_sizes)

        #corners and shape based off this image: https://www.researchgate.net/profile/Michael-Higgins-3/publication/31264562/figure/fig5/AS:650757183389706@1532164011563/Schematic-illustration-of-the-response-of-plagioclase-crystal-faces-to-the-chemical.png
        #here we look at the facetted faces as a fixed proportion of lengths

        s = self.s
        i = self.i
        l = self.l
        i_int = (1-2*del_i)*i
        l_int = (1-2*del_l)*l
        delta = (del_i*del_l*l)/(0.5-del_i)

        self.corners = np.asarray([[0,0.5*i,0], [s, 0.5*i, 0], [0,del_i*i, del_l*l], [s, del_i*i, del_l*l], [0.5*s,0,del_l*l+delta], [0.5*s,0,l-del_l*l-delta], 
                        [0,del_i*i, l-del_l*l], [s, del_i*i, l-del_l*l], [0,0.5*i,l], [s, 0.5*i, l], [0,i-del_i*i, l-del_l*l], [s, i-del_i*i, l-del_l*l],
                        [0.5*s,i,l-del_l*l-delta], [0.5*s,i,del_l*l+delta], [0,i - del_i*i, del_l*l], [s, i - del_i*i, del_l*l]])
        self.connections = np.asarray([[0,1], [0,2], [1,3], [2,4], [2,6], [3,4], [3,7],[4,5], [5,6], [5,7], [6,8],[7,9],[8,9], [8,10], [9,11], [10,12],[10,14],
                            [11,12], [11,15],[12,13], [13,14], [13,15], [14,0], [15,1]])
        
