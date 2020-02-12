# xwp_rotation

Utilities for rotation of discretized 3D objects, to be used to generate rotation matrices for use by other projects. 

- #### xwp_rotation 
  - *utils*     - `get_valid_locs` to determine safe filling area
  - *rotate*    - `pyvips_rotate` to rotate 2D/3D numpy arrays, `rot_matrix` to generate rotation matrix
  - *conv_dict* - conversion dictionary between numpy/pyvips
- #### tests
  - *basic*   - demonstration of rotation matrix 
  - *phantom* - demonstration of rotation of one slice of phantom using rotation matrix in PETSc.


The phantom test case conatins data from [EMD-3756](https://www.emdataresource.org/EMD-3756).
