import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


setup(
    name='mydetector3d',
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[make_cuda_ext(
                name="bev_pool_ext",
                module="mydetector3d.ops.bev_pool",
                sources=[
                    "src/bev_pool.cpp",
                    "src/bev_pool_cuda.cu",
                    ],
                )
        ]

)