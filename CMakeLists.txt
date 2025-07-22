cmake_minimum_required(VERSION 3.15)
project(3dba LANGUAGES CXX)

# ─── C++ Standard ───────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options ─────────────────────────────────────
option(USE_WANDB "Enable Weights & Biases logging" OFF)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/install/ceres-1.14" ${CMAKE_PREFIX_PATH}) #Local installation of Ceres 1.14 for avatar

# ─── FetchContent (needed only for W&B) ─────────
include(FetchContent)

# ─── Find system packages ───────────────────────
find_package(Eigen3       3.3 REQUIRED NO_MODULE)
find_package(Ceres  REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(pybind11 CONFIG REQUIRED)

# ─── Configure avatar  ──────────────────────────
set(WITH_PCL ON CACHE BOOL "Enable PCL support")
set(WITH_K4A OFF CACHE BOOL "" FORCE)
set(WITH_FREENECT2 OFF CACHE BOOL "" FORCE)
set(WITH_OMP ON CACHE BOOL "Enable OpenMP support")
add_subdirectory(external/avatar EXCLUDE_FROM_ALL)

# ─── Main executable ────────────────────────────
add_executable(3dba
    src/main.cpp
)

target_include_directories(3dba PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/avatar/include
    ${CMAKE_CURRENT_BINARY_DIR}/external/avatar/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(3dba PRIVATE
    avatar
    ceres
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
)

# ─── Link W&B only if enabled ───────────────────
if(USE_WANDB)
    target_link_libraries(3dba PRIVATE wandbcpp)
    target_compile_definitions(3dba PRIVATE USE_WANDB)
endif()
