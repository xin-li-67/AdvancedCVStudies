project(ncnn_yolov6)

cmake_minimum_required(VERSION 3.10)

find_package(OPENCV REQUIRED)

set(ncnn_DIR ${CMAKE_SOURCE_DIR})
find_package(ncnn REQUIRED)

add_library(yolov6 SHARED yolov6.cpp)

target_link_libraries(yolov6 ncnn ${OpenCV_LIBS})