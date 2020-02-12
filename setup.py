from setuptools import setup

setup(
        name             = 'xwp_rotation',
        description      = 'Utilities for 2D/3D object rotation',
        author           = 'Sajid Ali',
        author_email     = 'sajidsyed2021@u.northwestern.edu',
        packages         = ['xwp_rotation'],
        install_requires = ['numpy', 'matplotlib', 'pyvips']
)
