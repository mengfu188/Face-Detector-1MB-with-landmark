cmake_minimum_required(VERSION 2.8)

project(FaceDetector)

add_definitions(-std=c++11)
add_definitions("-Wall")

# set(OpenCV_DIR "/usr/local/opencv/")
find_package(OpenCV REQUIRED PATHS /home/cmf/dev/opencv-4.2.0/build_extra/install)

#if(True)
#    find_package(OpenMP)
#    if(NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
#        message(NOT TARGET OpenMP::OpenMP_CXX AND)
#        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#    else()
#        message(TARGET OpenMP::OpenMP_CXX AND)
#    endif()
#endif()

include_directories(${OpenCV_INCLUDE_DIRS}
        /home/cmf/dev/ncnn/build/install/include/ncnn)

link_directories(${OpenCV_LIBS}
        /home/cmf/dev/ncnn/build/install/lib)
#set(OpenCV_LIBS opencv_core
#                opencv_highgui
#                opencv_imgproc
#                opencv_imgcodecs
#                )

set(SRC
        ssd_on_image.cpp
        FaceDetector.h
        FaceDetector.cpp
        )

#add_executable(FaceAlive FaceAlive.cpp FaceDetector.cpp FaceDetector.h)
#target_link_libraries(FaceAlive ncnn ${OpenCV_LIBS})

add_executable(ImageFaceDetector ${SRC})
target_link_libraries(ImageFaceDetector ncnn ${OpenCV_LIBS})

add_executable(VideoFaceDetector FaceDetector.h FaceDetector.cpp ssd_on_video.cpp)
target_link_libraries(VideoFaceDetector ncnn ${OpenCV_LIBS})

add_executable(test_c test_c.cpp)
target_link_libraries(test_c ncnn ${OpenCV_LIBS})

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS})

add_executable(track track.cpp optical_tracker.cpp optical_tracker.h FaceDetector.h FaceDetector.cpp)
target_link_libraries(track ncnn ${OpenCV_LIBS})

add_executable(recognize face_recognize.cpp FaceDetector.h FaceDetector.cpp)
target_link_libraries(recognize ncnn ${OpenCV_LIBS})

add_executable(simple_run simple_run.cpp)
target_link_libraries(simple_run ncnn ${OpenCV_LIBS})

add_executable(face_track face_track.cpp)
target_link_libraries(face_track ncnn ${OpenCV_LIBS})

add_executable(track_demo track_demo.cpp)
target_link_libraries(track_demo ncnn ${OpenCV_LIBS})

add_executable(track_detect track_detect.cpp FaceDetector.h FaceDetector.cpp)
target_link_libraries(track_detect ncnn ${OpenCV_LIBS})

add_executable(mtcnn_demo mtcnn_demo.cpp mtcnn.cpp mtcnn.h base.cpp base.h)
target_link_libraries(mtcnn_demo ncnn ${OpenCV_LIBS})

if (OpenMP_CXX_FOUND)
    if (NCNN_CMAKE_VERBOSE)
        message("Building with OpenMP")
    endif ()
    target_link_libraries(ImageFaceDetector PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(VideoFaceDetector PUBLIC OpenMP::OpenMP_CXX)
endif ()
