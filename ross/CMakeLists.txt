cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 17)

project(ross-testing)

add_executable(ross-test main.cpp)

find_package(OpenCV CONFIG REQUIRED)
find_library(OpenCV NAMES OpenCV)

target_link_libraries ( ross-test PRIVATE ${OpenCV_LIBS})

install(TARGETS ross-test DESTINATION bin)
