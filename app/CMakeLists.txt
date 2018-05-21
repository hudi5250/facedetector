cmake_minimum_required(VERSION 3.4.1)



add_library(
             native-lib
             SHARED
             src/main/cpp/native-lib.cpp
             src/main/cpp/boundingbox.h
             src/main/cpp/caffe_sane.h
             src/main/cpp/caffe_sane.cpp
             src/main/cpp/calcarea.h
             src/main/cpp/calcarea.cpp
             src/main/cpp/Facedetection.h
             )
find_library(
          android-lib
          android
          )

include(AndroidNdkModules)
android_ndk_import_module_cpufeatures()

add_library(
    caffe2
    STATIC
    IMPORTED
    )
set_target_properties(
    caffe2
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libCaffe2_CPU.a
    )
add_library(
    thread_pool
    STATIC
    IMPORTED
    )
set_target_properties(
    thread_pool
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libCAFFE2_PTHREADPOOL.a
    )
add_library(
    glog
    SHARED
    IMPORTED
    )
set_target_properties(
    glog
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libglog.so
    )

add_library(
    protobuf
    SHARED
    IMPORTED
    )
set_target_properties(
    protobuf
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libprotobuf.a
    )

add_library(
    NNPACK
    STATIC
    IMPORTED
    )
set_target_properties(
    NNPACK
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libCAFFE2_NNPACK.a
    )

include_directories( src/main/cpp )

find_library(
     log-lib
     log
     )

add_library(lib_opencv SHARED IMPORTED)

set_target_properties(
    lib_opencv
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_java3.so
    )

target_link_libraries(
                       native-lib
                       -Wl,--whole-archive
                       caffe2
                       -Wl,--no-whole-archive
                       NNPACK

                       thread_pool
                       glog
                       protobuf
                       cpufeatures
                       ${log-lib}
                       ${android-lib}
                       lib_opencv)