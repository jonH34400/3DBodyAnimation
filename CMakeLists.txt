cmake_minimum_required(VERSION 3.14)
project(3DBodyAnimation)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(src)

add_executable(3DBodyAnimation
    src/main.cpp
    src/optimizer.cpp
    src/smpl_model.cpp
    src/visualizer.cpp
)

target_link_libraries(3DBodyAnimation
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)