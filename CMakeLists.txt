cmake_minimum_required(VERSION 3.15)
project(3dba LANGUAGES CXX)

# ─── C++ Standard ──────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options ───────────────────────────────────────────────────────────────
option(USE_WANDB "Enable Weights & Biases logging" OFF)

#  (local Ceres install path, keep your line)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/install/ceres-1.14"
                      ${CMAKE_PREFIX_PATH})

# ─── Dependencies ─────────────────────────────────────────────────────────
find_package(Eigen3       3.3 REQUIRED NO_MODULE)
find_package(Ceres        REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenCV       REQUIRED)

# ─── Avatar library ───────────────────────────────────────────────────────
set(WITH_PCL ON  CACHE BOOL "")
set(WITH_K4A OFF CACHE BOOL "" FORCE)
set(WITH_FREENECT2 OFF CACHE BOOL "" FORCE)
set(WITH_OMP ON  CACHE BOOL "")
add_subdirectory(external/avatar EXCLUDE_FROM_ALL)

# convenience include list (avoids repeating)
set(PROJECT_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/avatar/include
    ${CMAKE_CURRENT_BINARY_DIR}/external/avatar/include
    ${OpenCV_INCLUDE_DIRS}
)

set(PROJECT_LIBS
    avatar
    ceres
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
)

# ─── Executable 1 : single frame  ──────────────────────────────────────────
add_executable(3dba_single  src/main_single_frame.cpp)

target_include_directories(3dba_single PRIVATE ${PROJECT_INCLUDES})
target_link_libraries     (3dba_single PRIVATE ${PROJECT_LIBS})

# optional W&B
if(USE_WANDB)
    target_link_libraries(3dba_single PRIVATE wandbcpp)
    target_compile_definitions(3dba_single PRIVATE USE_WANDB)
endif()

# ─── Executable 2 : multiframe  ────────────────────────────────────────────
### NEW ###
add_executable(3dba_multi   src/main_multi_frame.cpp)

target_include_directories(3dba_multi PRIVATE ${PROJECT_INCLUDES})
target_link_libraries     (3dba_multi PRIVATE ${PROJECT_LIBS})

if(USE_WANDB)
    target_link_libraries(3dba_multi  PRIVATE wandbcpp)
    target_compile_definitions(3dba_multi PRIVATE USE_WANDB)
endif()
### NEW ###

# ─── install targets (optional) ────────────────────────────────────────────
# install(TARGETS 3dba_single 3dba_multi RUNTIME DESTINATION bin)
