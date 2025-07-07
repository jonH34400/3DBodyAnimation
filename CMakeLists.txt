cmake_minimum_required(VERSION 3.15)
project(3dba LANGUAGES CXX)

find_package(Torch REQUIRED)          # add this once
find_package(Eigen3 3.3 REQUIRED NO_MODULE)   # header-only, no link step
# ── header-only xtl ─────────────────────────────────────────────────────
add_library(xtl INTERFACE)
target_include_directories(xtl INTERFACE
    ${CMAKE_SOURCE_DIR}/external/xtl/include)

# ── header-only xtensor (depends on xtl) ────────────────────────────────
set(XTENSOR_INC ${CMAKE_SOURCE_DIR}/external/xtensor/include)
add_library(xtensor INTERFACE)
target_include_directories(xtensor INTERFACE ${XTENSOR_INC})
target_link_libraries(xtensor INTERFACE xtl)
set(xtensor_INCLUDE_DIRS ${XTENSOR_INC})         # for legacy CMake in SMPLpp

# ── SMPLpp library (uses our xtensor target) ────────────────────────────
add_subdirectory(external/SMPLpp)

# ── optimisation pipeline executable ────────────────────────────────────
add_executable(3dba
    src/main.cpp
)
target_link_libraries(3dba
    PRIVATE smplpp ${TORCH_LIBRARIES} xtensor stdc++fs Eigen3::Eigen
)
target_compile_features(3dba PRIVATE cxx_std_17)

add_executable(single_frame_fit
    src/demo_single_frame.cpp
)
target_link_libraries(single_frame_fit
    PRIVATE smplpp Eigen3::Eigen ${TORCH_LIBRARIES} xtensor ceres
)
target_compile_features(single_frame_fit PRIVATE cxx_std_17)
