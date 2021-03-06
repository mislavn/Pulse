CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(Pulse VERSION 0.0.1 LANGUAGES CXX)

find_program(CCACHE ccache)
if(CCACHE)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
endif()

# Link this 'library' to use the standard warnings
add_library(project_warnings INTERFACE)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)

target_compile_features(project_options INTERFACE cxx_std_17)

#TODO add static compile
#SET(BUILD_SHARED_LIBS OFF)
#SET(CMAKE_EXE_LINKER_FLAGS "-static")

target_compile_options(project_warnings
INTERFACE -Wall
          -Werror # treat all warnings as errors
          -Wextra # reasonable and standard
          -Wshadow # warn the user if a variable declaration shadows one
                   # from a parent context
          -Wnon-virtual-dtor # warn the user if a class with virtual
                             # functions has a non-virtual destructor. This
                             # helps catch hard to track down memory errors
          -Wold-style-cast # warn for c-style casts
          -Wcast-align # warn for potential performance problem casts
          -Wunused # warn on anything being unused
          -Woverloaded-virtual # warn if you overload (not override) a
                               # virtual function
          -Wpedantic # warn if non-standard C++ is used
          -Wconversion # warn on type conversions that may lose data
          -Wsign-conversion # warn on sign conversions
          -Wnull-dereference # warn if a null dereference is detected
          -Wdouble-promotion # warn if float is implicit promoted to double
          -Wformat=2 # warn on security issues around functions that format
                     # output (ie printf)
)

option(ENABLE_CLANG_TIDY "Enable testing with clang-tidy" TRUE)
option(ENABLE_CPPCHECK "Enable testing with cppcheck" TRUE)

if(ENABLE_CPPCHECK)
    find_program(CPPCHECK cppcheck)
    if(CPPCHECK)
        set(CMAKE_CXX_CPPCHECK ${CPPCHECK} --suppress=syntaxError --enable=all --inconclusive)
    else()
        message(SEND_ERROR "cppcheck requested but executable not found")
    endif()
endif()

if(ENABLE_CLANG_TIDY)
    find_program(CLANGTIDY clang-tidy)
    if(CLANGTIDY)
        set(CMAKE_CXX_CLANG_TIDY ${CLANGTIDY})
    else()
        message(SEND_ERROR "clang-tidy requested but executable not found")
    endif()
endif()

# Download automatically, you can also just copy the conan.cmake file
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
   message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
   file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.13/conan.cmake" "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)
conan_check(REQUIRED)
conan_cmake_run(CONANFILE conanfile.txt BASIC_SETUP CONAN_COMMAND ${CONAN_CMD} CMAKE_TARGETS BUILD missing)

set(SOURCES
	src/webcam.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

# generate file with path to casscade classifiers
set(HAARCASCADES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/haarcascades/haarcascade_frontalface_default.xml" CACHE STRING "Path to haarcascades XML file.")
configure_file("${PROJECT_SOURCE_DIR}/config/config.hpp.in" "${PROJECT_SOURCE_DIR}/src/config.hpp" ESCAPE_QUOTES @ONLY)

# dependencies - OpenCV
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE project_options project_warnings CONAN_PKG::opencv)
