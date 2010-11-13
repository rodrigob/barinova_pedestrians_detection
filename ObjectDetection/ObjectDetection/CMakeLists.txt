# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# to compile the local code you can use: cmake ./ && make -j2

# ----------------------------------------------------------------------
# Base CMake setup

cmake_minimum_required (VERSION 2.6)

# ----------------------------------------------------------------------
# Setup the project

include(FindPkgConfig)
project (ObjectDetection)

# ----------------------------------------------------------------------
# Setup require libraries
#pkg_check_modules(libpng REQUIRED libpng)
pkg_check_modules(opencv REQUIRED opencv>=2.1)
#set(vw_LIBRARIES "-lvwCore -lvwImage -lvwStereo -lvwFileIO -lvwMath -lvwInterestPoint")

# ----------------------------------------------------------------------
# Setup link and include directories

set(local_LIBRARY_DIRS
  "/users/visics/rbenenso/no_backup/usr/local/lib"
  "/usr/lib64"
  "/usr/lib64/atlas"
  "/usr/lib/sse2/atlas"
)
set(local_INCLUDE_DIRS
  "/users/visics/rbenenso/no_backup/usr/local/include"
  "/usr/include/eigen2/"
   "/usr/local/include/eigen2"
)


link_directories(
	${libpng_LIBRARY_DIRS} 
	${opencv_LIBRARY_DIRS}
)
include_directories(
	"../Utils/" 
	"${biclop_root}/src" 
	${libpng_INCLUDE_DIRS} 
	${opencv_INCLUDE_DIRS} 
)

# ----------------------------------------------------------------------
# Collect source files


file(GLOB SrcCpp 
	"../DenseGreedyDetection/*.cpp"
	"../Forests/*.cpp"
	"../ImageWrapper/*.cpp"
	"../Utils/*.cpp"
)


# ----------------------------------------------------------------------
# Create the executable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -p ${OPT_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -DNDEBUG ${OPT_CXX_FLAGS}")

add_executable(object_detection ${SrcCpp})

target_link_libraries(object_detection

   ${cg_LIBRARIES}
   # linking with CgGL _after_ boost_program_options generates a segmentation fault ! boost_program_options 1.39 has a bug

   boost_program_options-mt boost_filesystem-mt boost_system-mt boost_thread-mt protobuf
   SDL
   gomp Xext Xrandr X11
   ${libpng_LIBRARIES} 
   ${opencv_LIBRARIES}

   #csparse sparse spblas mv
   lapack blas atlas
)   
 
# ----------------------------------------------------------------------
