cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 11)

project(yolov5dnn)

find_package(OpenCV REQUIRED)

add_executable(yolov5dnn yolov5.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov5dnn ${OpenCV_LIBS} ${CAMKE_THREAD_LIBS_INIT})