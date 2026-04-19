cmake_minimum_required(VERSION 3.20)
project(rocket_integrator LANGUAGES CXX)

# ── C++ standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)      # no GNU extensions — pure ISO C++

# Default to Release if nothing is specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ── Fetch Eigen (header-only, no install required) ───────────────────────────
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
)
# Suppress Eigen's own tests and install targets
set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(rocket_sim src/main.cpp)

target_include_directories(rocket_sim PRIVATE include)
target_link_libraries(rocket_sim PRIVATE Eigen3::Eigen)

# ── Optimisation flags ────────────────────────────────────────────────────────
#
#   GCC / Clang:
#     -O3            — full optimisation (auto-vectorisation, inlining, loop unrolling)
#     -march=native  — use every SIMD instruction available on this machine
#     -DNDEBUG       — disable assert() and other debug-only checks
#     -DEIGEN_NO_DEBUG — disable Eigen's internal assertion checks (safe in release)
#
#   MSVC:
#     /O2            — maximise speed
#     /arch:AVX2     — enable AVX2 SIMD (adjust to /arch:AVX if CPU is older)
#     /GL            — whole-program optimisation
#     /DNDEBUG       — same as above
#
target_compile_options(rocket_sim PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
        -O3
        -march=native
        -DNDEBUG
        -DEIGEN_NO_DEBUG
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /O2
        /arch:AVX2
        /GL
        /DNDEBUG
        /DEIGEN_NO_DEBUG
    >
)

# Whole-program optimisation linker flag for MSVC
target_link_options(rocket_sim PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/LTCG>
)
