CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(Pulse)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")


set(SOURCES
	src/webcam.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

# generate file with path to casscade classifiers
set(HAARCASCADES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/haarcascades/haarcascade_frontalface_default.xml" CACHE STRING "Path to haarcascades XML file.")
configure_file("${PROJECT_SOURCE_DIR}/config/config.hpp.in" "${PROJECT_SOURCE_DIR}/src/config.hpp" ESCAPE_QUOTES @ONLY)

# dependencies - OpenCV
find_package(OpenCV REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} opencv_face opencv_core)
