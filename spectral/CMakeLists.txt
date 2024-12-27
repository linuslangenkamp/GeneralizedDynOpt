cmake_minimum_required(VERSION 3.10)
project(GDOPT_EXPERIMENTAL)

set(CMAKE_CXX_STANDARD 17)

file(GLOB SOURCES "src/*.cpp")

add_executable(gdopt_experimental ${SOURCES})
