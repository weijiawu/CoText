from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="kprocess",
    version="0.1.0",
    description="Using NNP warp_perspective",
    author="itsliupeng",
    classifiers=["Programming Language :: Python :: 3"],
    ext_modules=[
        # CUDAExtension(
        #     "kprocess",
        #     ["npp_ops.cc", "register_torch_ops.cpp"],
        #     extra_compile_args=['-std=c++14'],
        #     include_dirs=["/usr/local/include/torchvision"],
        #     libraries=["nppc", "nppig", "nppif", "torchvision"],
        #     define_macros=[('PYBIND', None)]
        # ),
        CUDAExtension(
            "kprocess",
            ["cuda_ops.cpp", "cuda/kernel.cu"],
            extra_compile_args=['-std=c++14'],
            include_dirs=[os.path.join(this_dir, "include"), this_dir],
            libraries=["nppc", "nppig", "nppif"],
            define_macros=[('PYBIND', None)]
        ),
        # CUDAExtension(
        #     "npp_ops",
        #     ["opencv_cuda_warp_perspective.cc"],
        #     extra_compile_args=['-std=c++14'],
        #     include_dirs=["/usr/local/include/opencv4"],
        #     libraries=["opencv_core", "opencv_imgproc", "opencv_cudawarping"],
        #     define_macros=[('PYBIND', None)]
        # )
    ],
    cmdclass={"build_ext": BuildExtension},
)
