cmake_minimum_required(VERSION 3.10)
project(WirelessMarlBeamforming CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /Ob2 /Oi /Ot /fp:fast /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra -pedantic)
endif()

include_directories(include)

add_executable(beamforming_benchmark src/main_benchmark.cpp)
