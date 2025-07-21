cmake_minimum_required(VERSION 3.15)
project(3dba LANGUAGES CXX)

# ─── C++ Standard ───────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options ─────────────────────────────────────
option(USE_WANDB "Enable Weights & Biases logging" OFF)

# ─── FetchContent (needed only for W&B) ─────────
include(FetchContent)

# ─── Find system packages ───────────────────────
find_package(Eigen3       3.3 REQUIRED NO_MODULE)
find_package(Ceres             REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ─── Configure smplxpp ──────────────────────────
set(SMPLX_BUILD_VIEWER OFF CACHE BOOL "" FORCE)
set(SMPLX_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory(external/smplxpp EXCLUDE_FROM_ALL)

# ─── Optionally Fetch W&B ───────────────────────
if(USE_WANDB)
    FetchContent_Declare(
        wandb_cpp
        GIT_REPOSITORY https://github.com/yhisaki/wandb-cpp.git
        GIT_TAG        master
    )
    FetchContent_MakeAvailable(wandb_cpp)

    set_target_properties(wandbcpp PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
    )
endif()

# ─── Main executable ────────────────────────────
add_executable(3dba
    src/main.cpp
    src/OptimizeSMPL.cpp
)

target_include_directories(3dba PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/smplxpp/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(3dba PRIVATE
    smplxpp
    ceres
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    pybind11::embed
)

# ─── Link W&B only if enabled ───────────────────
if(USE_WANDB)
    target_link_libraries(3dba PRIVATE wandbcpp)
    target_compile_definitions(3dba PRIVATE USE_WANDB)
endif()
