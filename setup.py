import os
import subprocess

from setuptools import find_packages, setup
#from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# def make_cuda_ext(name, module, sources):
#     cuda_ext = CUDAExtension(
#         name='%s.%s' % (module, name),
#         sources=[os.path.join(*module.split('.'), src) for src in sources]
#     )
#     return cuda_ext

if __name__ == '__main__':
    version = '0.1' #% get_git_commit_number()
    #write_version_to_file(version, 'pcdet/version.py')

    setup(
        name='mydetector3d',
        version=version,
        description='mydetector3d is a 3D object detection from point cloud',
        install_requires=[
            'numpy',
            'llvmlite',
            #'numba',
            'tensorboardX',
            'easydict',
            'pyyaml',
            'scikit-image',
            'tqdm',
            #'SharedArray', #https://pypi.org/project/SharedArray/
            # 'spconv',  # installed independently 
        ],

        author='Kaikai Liu',
        author_email='kaikai.liu@sjsu.edu',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        # cmdclass={
        #     'build_ext': BuildExtension,
        # },
        # ext_modules=[
            
        # ],
    )