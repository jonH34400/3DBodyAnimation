cmake_minimum_required(VERSION 3.15)
project(3dba LANGUAGES CXX)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)   

add_library(xtl INTERFACE)
target_include_directories(xtl INTERFACE
    ${CMAKE_SOURCE_DIR}/external/xtl/include)

set(XTENSOR_INC ${CMAKE_SOURCE_DIR}/external/xtensor/include)
add_library(xtensor INTERFACE)
target_include_directories(xtensor INTERFACE ${XTENSOR_INC})
target_link_libraries(xtensor INTERFACE xtl)
set(xtensor_INCLUDE_DIRS ${XTENSOR_INC})        

add_subdirectory(external/SMPLpp)

# main
add_executable(3dba
    src/main.cpp
)

target_link_libraries(3dba
    PRIVATE smplpp stdc++fs Eigen3::Eigen
)
target_compile_features(3dba PRIVATE cxx_std_17)

# single frame
# add_executable(single_frame_fit
#     src/demo_single_frame.cpp
# )
# target_link_libraries(single_frame_fit
#     PRIVATE smplpp Eigen3::Eigen ceres
# )
# target_compile_features(single_frame_fit PRIVATE cxx_std_17)
