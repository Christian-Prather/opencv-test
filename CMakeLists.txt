cmake_minimum_required(VERSION 3.2)

project(cl-tseting)

add_executable(umat-tests umat-test.cpp)

find_package(OpenCV CONFIG REQUIRED)
find_library(OpenCV NAMES OpenCV)

target_link_libraries ( umat-tests PRIVATE ${OpenCV_LIBS})

install(TARGETS umat-tests DESTINATION bin)