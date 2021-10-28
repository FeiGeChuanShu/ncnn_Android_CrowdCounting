project(p2pnetncnn)

cmake_minimum_required(VERSION 3.4.1)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.1-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20210720-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(p2pnetncnn SHARED p2pnet_ncnn.cpp)

target_link_libraries(p2pnetncnn ncnn ${OpenCV_LIBS} jnigraphics)
