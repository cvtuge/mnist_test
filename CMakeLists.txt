#要求CMake的最低版本为3.5  
cmake_minimum_required(VERSION 3.5)
#项目名字
project(mnist_test)
#c++标准
set(CMAKE_CXX_STANDARD 11)
#添加CPU_ONLY宏定义，只在CPU模式下运行
ADD_DEFINITIONS(-DCPU_ONLY)
#include_directories(include)
#依赖库路径
link_directories("/home/zch/Desktop/caffe-master/lib")
#头文件目录
include_directories("/home/zch/Desktop/caffe-master/include")
link_directories("/usr/local/cuda/lib64")
#指定源代码
set(SOURCE_FILES main.cpp)
#生成目标文件
add_executable(mnist_test ${SOURCE_FILES})
#查找opencv链接库路径
find_package(OpenCV REQUIRED)
#链接库文件
target_link_libraries(mnist_test ${OpenCV_LIBS} )
target_link_libraries(mnist_test caffe boost_system glog gflags)

