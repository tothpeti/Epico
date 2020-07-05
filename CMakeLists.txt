cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(epico-app)

# Set c++ version to C++14
set(CMAKE_CXX_STANDARD 14)

# Libtorch library
find_package(torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# The compiled library code is here
add_subdirectory(src)

# The executable code is here
add_subdirectory(app)
