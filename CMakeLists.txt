project(lloe)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Set up compiler flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -pedantic -Werror -Wall -Wextra -Wno-missing-braces -Wno-pragmas -Wno-unknown-pragmas")

# Set up include and library paths
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/extern)
include_directories("/usr/local/include")
link_directories("/usr/local/lib")

# Set up version number and main project configuration details
set(OGT_VERSION "0.1.0")
set(OGT_NAMESPACE "ogt")

# TODO: Set up testing
# enable_testing()
# add_subdirectory("tests")

# Find dlib and its dependencies
find_package(BLAS REQUIRED)
link_libraries(${BLAS_LIBRARIES})

find_package(LAPACK REQUIRED)
link_libraries(${LAPACK_LIBRARIES})

find_package(dlib REQUIRED)
include_directories(${DLib_INCLUDE_DIRS})
link_libraries(${DLib_LIBRARY})

# Find boost and its dependencies
find_package(Boost REQUIRED)
include_directories(${boost_INCLUDE_DIRS})
link_libraries("boost_iostreams")

# Find compression libraries
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)

# Generate the configuration header
configure_file ("${PROJECT_SOURCE_DIR}/config/config.hpp.tpl"
  "${PROJECT_BINARY_DIR}/include/ogt/config.hpp")

# Configure output binaries
add_subdirectory(src)

# Configure package settings
set(CPACK_PACKAGE_NAME "lloe")
set(CPACK_PACKAGE_VENDOR "Jesse Anderton")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "lloe - Large-Scale Landmark Ordinal Embedding")
string(REGEX REPLACE "([^.]*)\\.([^.]*)\\.([^.]*)" "\\1" CPACK_PACKAGE_VERSION_MAJOR "${OGT_VERSION}")
string(REGEX REPLACE "([^.]*)\\.([^.]*)\\.([^.]*)" "\\2" CPACK_PACKAGE_VERSION_MINOR "${OGT_VERSION}")
string(REGEX REPLACE "([^.]*)\\.([^.]*)\\.([^.]*)" "\\3" CPACK_PACKAGE_VERSION_PATCH "${OGT_VERSION}")
set(CPACK_PACKAGE_VERSION "${OGT_VERSION}")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "lloe")
