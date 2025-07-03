cmake_minimum_required(VERSION 3.15)
project(3dba LANGUAGES CXX)

# ── LibTorch (module mode) ──────────────────────────────────────────────
find_package(Torch REQUIRED)   

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
add_executable(3dba src/main.cpp)
target_include_directories(3dba PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(3dba
    PRIVATE smplpp xtensor ${TORCH_LIBRARIES}
)
target_compile_features(3dba PRIVATE cxx_std_17)

# ── smoke-test executable ───────────────────────────────────────────────
add_executable(smpl_test tests/test_smpl_neutral.cpp)
target_include_directories(smpl_test PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(smpl_test
    PRIVATE smplpp xtensor ${TORCH_LIBRARIES}
)
target_compile_features(smpl_test PRIVATE cxx_std_17)
