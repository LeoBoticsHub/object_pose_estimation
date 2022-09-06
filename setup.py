from setuptools import setup

package_name = 'object_pose_estimation'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[],
    install_requires=['setuptools', 'numpy', 'open3d', 'opencv-python', 'matplotlib'],
    zip_safe=True,
    maintainer='fabio-amadio',
    maintainer_email='fabioamadio93@gmail.com',
    description='This package contains functions useful for camera use and other computer vision computations',
    license='GNU GENERAL PUBLIC LICENSE v3',
    entry_points={},
)
