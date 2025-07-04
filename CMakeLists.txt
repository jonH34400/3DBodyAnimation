cmake_minimum_required(VERSION 3.15)
project(3dba LANGUAGES CXX)

find_package(Torch REQUIRED)          # add this once

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
    PRIVATE smplpp ${TORCH_LIBRARIES} xtensor stdc++fs
)
target_compile_features(3dba PRIVATE cxx_std_17)

# ── smoke-test executable ───────────────────────────────────────────────
# add_executable(smpl_test
#     tests/test_smpl_neutral.cpp
# )
# target_link_libraries(smpl_test
#     PRIVATE smplpp torch xtensor
# )
# target_compile_features(smpl_test PRIVATE cxx_std_17)