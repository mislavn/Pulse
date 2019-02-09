CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(Pulse VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

find_program(CCACHE ccache)
if(CCACHE)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
endif()

# Link this 'library' to use the standard warnings
add_library(project_warnings INTERFACE)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)

target_compile_features(project_options INTERFACE cxx_std_17)
target_include_directories(project_options INTERFACE include external)

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

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(project_warnings
    INTERFACE -Wmisleading-indentation # warn if identation implies blocks
                                     # where blocks do not exist
              -Wduplicated-cond # warn if if / else chain has duplicated
                                # conditions
              -Wduplicated-branches # warn if if / else branches have
                                    # duplicated code
              -Wlogical-op # warn about logical operations being used where
                           # bitwise were probably wanted
              -Wuseless-cast # warn if you perform a cast to the same type
    )
    target_link_libraries(project_options INTERFACE stdc++fs)
else()
    target_link_libraries(project_options INTERFACE stdc++fs)
endif()

option(ENABLE_CLANG_TIDY "Enable testing with clang-tidy" FALSE)
option(ENABLE_CPPCHECK "Enable testing with cppcheck" FALSE)

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

set(SOURCES
	src/webcam.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

# generate file with path to casscade classifiers
set(HAARCASCADES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/haarcascades/haarcascade_frontalface_default.xml" CACHE STRING "Path to haarcascades XML file.")
configure_file("${PROJECT_SOURCE_DIR}/config/config.hpp.in" "${PROJECT_SOURCE_DIR}/src/config.hpp" ESCAPE_QUOTES @ONLY)

# dependencies - OpenCV
find_package(OpenCV REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} opencv_face opencv_core)
