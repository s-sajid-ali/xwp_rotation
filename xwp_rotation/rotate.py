import numpy as np
import pyvips

from .conv_dict import format_to_dtype, dtype_to_format
from .utils import get_valid_locs

__all__ = ['pyvips_rotate', 'rot_matrix']


'''

Utility function to convert 2D/3D numpy arrays using pyvips. 
(Note that the numpy arrays must be c-contiguous.)

obj   : a 2D/3D numpy object to be rotated
angle : angle in degrees to rotate the object by.

'''

def pyvips_rotate(obj,angle):
    # convert angle to radians
    # get cos/sin 
    angle_rad = angle * (np.pi/180)
    tc = np.cos(angle_rad)
    ts = np.sin(angle_rad)
    
    # Set rows and columns for 2D transform
    # This would still work for a stack of 
    # 2D images
    rows, cols = obj.shape[0], obj.shape[1]

    # Set center of array to be the 
    # aobjis of rotation.
    c_ = np.array((rows,cols))/2 - 0.5

    # Check if we are working with a 
    # single 2D array or a stack of them
    if len(obj.shape)!=3 : 
        height, width = obj.shape
        bands = 1
    else :
        height, width,bands = obj.shape
    
    # Create a pyvips image from the numpy array
    im = pyvips.Image.new_from_memory(obj.reshape(width * height * bands).data, width, height, bands,
                                      dtype_to_format[str(obj.dtype)])

    # Specify the interpolation type to bilinear. 
    # this interpolation will be used for roataion.
    inter = pyvips.vinterpolate.Interpolate.new('bicubic')

    # Perform the rotation via the pyvips
    # affine transform. Use the aforementioned 
    # bilinear interpolator.
    im = im.affine([tc, ts, -ts, tc], interpolate = inter,
                   idx = -c_[1], idy = -c_[0],
                   odx =  c_[1], ody =  c_[0],
                   oarea = [0, 0, im.width, im.height])

    # Transfer the pyvips image to numpy
    b = np.ndarray(buffer=im.write_to_memory(),dtype=format_to_dtype[im.format],shape=[im.height, im.width, im.bands])
    
    # delete the pyvips image 
    del im

    # Reshape the data to be either a 2D
    # 3D array.
    if len(obj.shape)==3 : 
        b = b.reshape(np.shape(b)[0],np.shape(b)[1],np.shape(b)[2])    
    else :
        b = b.reshape(np.shape(b)[0],np.shape(b)[1])
    
    # Return the rotated data with correct 
    # dimensions.
    return b




'''

Generate the rotation matrix for a given angle and
grid size.

2D-image (say img) -> 1D-vector (say _img)
Rotation of 2D-iamge can be performed by :
rot_matrix @ _img -> _img*

grid_size : 2D image shape is grid_size x grid_size
angle     : angle in degrees
               
'''

def rot_matrix(grid_size, angle):
    
    # width for grid_size obtained 
    # by visual inspection using
    # xwp_rotation.utils.get_valid_locs
    
    if grid_size == 128:
        width = 60
    if grid_size == 64:
        width = 30
    
    a = np.zeros((grid_size,grid_size))
    _a = get_valid_locs(a,width,plot=0)

    _angle = -angle
    
    _matrix = np.zeros((grid_size*grid_size,grid_size*grid_size))
        
    for i in range(_a.shape[1]):
        idx,idy = _a[:,i]
        a[idx,idy] = 1
        j = np.where(a.ravel()==1)
        b = pyvips_rotate(a,-angle)
        _matrix[j,:] = b.ravel()
        a[idx,idy] = 0

    return _matrix