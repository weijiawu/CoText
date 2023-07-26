# 3.18 CUDA_nppicom_LIBRARY-NOTFOUND
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(kprocess LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message("Build Type: " ${CMAKE_BUILD_TYPE})
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Configure nvcc with '-g -G' flag")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif ()

#cuda
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
#set(CUDA_npp_LIBRARY "${CUDA_nppc_LIBRARY};${CUDA_nppial_LIBRARY};${CUDA_nppicc_LIBRARY};${CUDA_nppidei_LIBRARY};${CUDA_nppif_LIBRARY};${CUDA_nppig_LIBRARY};${CUDA_nppim_LIBRARY};${CUDA_nppist_LIBRARY};${CUDA_nppisu_LIBRARY};${CUDA_nppitc_LIBRARY}")
message("CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}, CUDA_LIBRARIES ${CUDA_LIBRARIES}, CUDA_nppc_LIBRARY ${CUDA_nppc_LIBRARY}, CUDA_nppig_LIBRARY ${CUDA_nppig_LIBRARY}, CUDA_nppif_LIBRARY${CUDA_nppif_LIBRARY}")
#cudnn
set(CUDNN_ROOT_DIR /usr/local/cudnn)
find_library(CUDNN_LIBRARY_PATH cudnn HINTS ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib64 lib)
set(CUDNN_INCLUDE_PATH ${CUDNN_ROOT_DIR}/include)
include_directories(${CUDNN_INCLUDE_PATH})
#get python path, libtorch
set(PYTHON_PATH "/home/liupeng/anaconda3/envs/py36/bin/python" CACHE STRING "Python path")
# python include dir
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; from distutils import sysconfig; print(sysconfig.get_python_inc(),end='');"
        RESULT_VARIABLE _PYTHON_SUCCESS
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS)
if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Python config error")
endif ()
# python libs
include_directories(${PYTHON_INCLUDE_DIRS})
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; from distutils import sysconfig; print(sysconfig.get_config_var('LIBDIR'),end='');"
        RESULT_VARIABLE _PYTHON_SUCCESS
        OUTPUT_VARIABLE PYTHON_LIBDIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Python config error")
endif ()
set(PYTHON_LIBRARIES ${PYTHON_LIBDIR}/libpython3.6m.so)
message("PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS}, PYTHON_LIBRARIES ${PYTHON_LIBRARIES}")


execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch; print(torch.__version__,end='');"
        RESULT_VARIABLE _PYTHON_SUCCESS
        OUTPUT_VARIABLE TORCH_VERSION)
if (TORCH_VERSION VERSION_LESS "1.7.0")
    message(FATAL_ERROR "PyTorch >= 1.7.0 is needed for TorchScript mode.")
endif ()
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import torch;print(os.path.dirname(torch.__file__),end='');"
        RESULT_VARIABLE _PYTHON_SUCCESS
        OUTPUT_VARIABLE TORCH_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Torch config Error.")
endif ()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; from distutils import sysconfig;
print(sysconfig.get_python_inc());"
        RESULT_VARIABLE _PYTHON_SUCCESS
        OUTPUT_VARIABLE PY_INCLUDE_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Python config Error.")
else ()
    message(STATUS "Find Torch VERSION ${Torch_VERSION}, TORCH_CXX_FLAGS ${TORCH_CXX_FLAGS} TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS}, TORCH_LIBRARIES ${TORCH_LIBRARIES}")
    # attention _GLIBCXX_USE_CXX11_ABI=0 in pytorch
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    include_directories(${TORCH_INCLUDE_DIRS})
endif ()

# opencv 4.5.2
find_package(OpenCV 4.5.2 REQUIRED)
message("OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_INSTALL_PATH}/lib)
message("OpenCV_INSTALL_PATH: " ${OpenCV_INSTALL_PATH})
set(CV_LIBS opencv_core opencv_imgproc opencv_imgcodecs
        opencv_cudafilters opencv_cudawarping opencv_cudafeatures2d opencv_cudaarithm opencv_cudaimgproc opencv_ximgproc)

#find_library(OPENCV_CUDA /Users/liupeng/code/kuaishou-nv-opencv-cuda/build/install)
#set(OPENCV_CUDA_DIR /Users/liupeng/code/kuaishou-nv-opencv-cuda/build/install)
#find_library(OPENCV_CUDA_LIBRARY opencv_cuda HINTS ${OPENCV_CUDA_DIR} PATH_SUFFIXES lib)
#set(OPENCV_CUDA_INCLUDE_PATH ${OPENCV_CUDA_DIR}/include)
#include_directories(${OPENCV_CUDA_INCLUDE_PATH})
#message("OPENCV_CUDA_INCLUDE_PATH ${OPENCV_CUDA_INCLUDE_PATH}, OPENCV_CUDA_LIBRARY ${OPENCV_CUDA_LIBRARY}")

include_directories(.)
include_directories(include)
add_subdirectory(cuda)

enable_testing()
add_subdirectory(test/unit)

## torchvision
#find_package(TorchVision REQUIRED)
#set(TorchVision_INCLUDE_DIR ${TorchVision_INCLUDE_DIR}/torchvision)
#set(TorchVision_LIBRARY /usr/local/lib/libtorchvision.so)
#message(STATUS "TorchVision_INCLUDE_DIR ${TorchVision_INCLUDE_DIR}, TorchVision_LIBRARY: ${TorchVision_LIBRARY}")

add_library(register_torch_ops register_torch_ops.cpp)
target_include_directories(register_torch_ops PRIVATE ${TorchVision_INCLUDE_DIR})
target_link_libraries(register_torch_ops PRIVATE ${TORCH_LIBRARIES} ${TorchVision_LIBRARY})

#add_executable(run_cuda_warp_perspective check/cuda_warp_perspective.cc)
#target_link_libraries(run_cuda_warp_perspective PRIVATE ${OPENCV_CUDA_LIBRARY} ${CV_LIBS} ${TORCH_LIBRARIES})
#add_executable(run_opencv_cuda_warp_perspective check/opencv_cuda_warp_perspective.cc)
#target_link_libraries(run_opencv_cuda_warp_perspective PRIVATE ${CV_LIBS} ${TORCH_LIBRARIES})

add_library(kprocess cuda_ops.cpp)
target_link_libraries(kprocess PUBLIC cuda_kernels ${CUDA_nppc_LIBRARY} ${CUDA_nppig_LIBRARY} ${CUDA_nppif_LIBRARY} ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
