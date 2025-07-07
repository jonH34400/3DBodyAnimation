cmake_minimum_required(VERSION 3.15)
project(3dba LANGUAGES CXX)

# ─── C++17 ──────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── find our deps ─────────────────────────────
find_package(Eigen3       3.3 REQUIRED NO_MODULE)
find_package(Ceres         REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

#set(SMPLXPP_MODEL "SMPL" CACHE STRING
 #   "Which SMPL(X) model to build: SMPL / SMPLH / SMPLX / SMPLXpca" FORCE)
set(SMPLX_BUILD_VIEWER OFF CACHE BOOL "" FORCE)
set(SMPLX_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

# This will configure & build libsmplxpp.a (target name = smplxpp)
add_subdirectory(external/smplxpp EXCLUDE_FROM_ALL)

# ─── your app ───────────────────────────────────
add_executable(3dba
    src/main.cpp
    src/OptimizeSMPL.cpp
)

# expose your own public headers
target_include_directories(3dba PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    # and still need their public headers too:
    ${CMAKE_CURRENT_SOURCE_DIR}/external/smplxpp/include
)

# ─── link everyone ──────────────────────────────
target_link_libraries(3dba PRIVATE
    smplxpp                     # the real library target from smplxpp/
    ceres
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)
