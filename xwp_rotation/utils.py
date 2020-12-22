import numpy as np
import matplotlib.pyplot as plt


__all__ = ['get_valid_locs']


'''
Santiy checking program to ensure that one only fills the 
a subset of the available space in an array such that any
rotation transform does not lead to parts of the object 
landing outside the total available space.

support_grid : a 2D grid representing the available space, 
               could just be a grid of 1's or 0's as this
               routine just used the shape and not the values
width        : distance from center of the grid to fill
plot         : plot the pixels that fill the available width
'''

def get_valid_locs(support_grid,width,plot=0):
    cx,cy = np.shape(support_grid)[0]//2,np.shape(support_grid)[1]//2
    nr,nc = np.shape(support_grid)
    r = np.arange(nr)-cx 
    c = np.arange(nc)-cy 
    [R,C] = np.meshgrid(r,c)
    index = np.round(np.sqrt(R**2+C**2))+1 
    valid_locs_1 = []
    valid_locs_2 = []
    temp = np.max(support_grid)

    for i in range(width):
        j = i
        if len(np.where(index==j)[0]):
            valid_locs_1.append(np.where(index==j)[0])
            valid_locs_2.append(np.where(index==j)[1])
        
    if plot==1 :
        plt.imshow((support_grid),alpha=0.05,origin='lower')
        for i in np.arange(-width,width):
            j = i 
            R1,C1 = np.where(index==j)
            plt.scatter(C1,R1,s=5)
        plt.show()
    
    valid_locs_1 = np.concatenate(np.array(valid_locs_1, dtype=object))
    valid_locs_2 = np.concatenate(np.array(valid_locs_2, dtype=object))
    
    return np.vstack((valid_locs_1,valid_locs_2))
