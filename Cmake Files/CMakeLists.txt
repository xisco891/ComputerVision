project (xisco)

cmake_minimum_required (VERSION 2.6)
set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)

SET(CMAKE_INSTALL_BINDIR $ENV{HOME}/bin)
SET(NDK_ROOT "$ENV{NDK}")


find_package( OpenCV 3 REQUIRED )
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#include_directories( ${OpenCV_INCLUDE_DIRS} )
MESSAGE(STATUS "The OPENCV path is ${OpenCV_INCLUDE_DIR}")


#----------------BOOST LIBRARY--------------------#
#set(Boost_INCLUDE_DIR /usr/local/src/boost_1_46_1)
#set(Boost_LIBRARY_DIR /usr/local/src/boost_1_46_1/stage/lib)
#find_package(Boost COMPONENTS system filesystem REQUIRED)
#include_directories(${Boost_INCLUDE_DIR})
#link_directories(${Boost_LIBRARY_DIR})



#----------------CUDA LIBRARY--------------------#
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
add_definitions(${CUDA_DEFINITIONS})
MESSAGE(STATUS "The CUDA path is ${CUDA_INCLUDE_DIR}")

#-------------GLUT LIBRARY----------------------#
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})

if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
MESSAGE(STATUS "The OPENGL path is ${OPENGL_INCLUDE_DIR}")

find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )
MESSAGE(STATUS "the eigen path is ${EIGEN_INCLUDE_DIR}")

include_directories("/usr/include/libusb-1.0/")
include_directories(${FREENECT_INCLUDE_DIR})


set(SDL_INCLUDE_DIR /usr/include/SDL)
set(SDL_LIBRARY /usr/lib/libSDL.so)


set(SDL2_INCLUDE_DIR /usr/include/SDL2)
set(SDL2_LIBRARY /usr/lib/libSDL2.so)
#find_package(SDL2)
#include_directories(${SDL2_INCLUDE_DIR})
MESSAGE(STATUS "the SDL2 path is ${SDL2_INCLUDE_DIR}")


find_package (OpenCV REQUIRED)
include_directories (${OpenCV_INCLUDE_DIR})
MESSAGE(STATUS "the opencv path is ${OpenCV_INCLUDE_DIR}")

#find_package(libfreenect REQUIRED)
#MESSAGE(STATUS "the freenect path is ${FREENECT_INCLUDE_DIR}")

#set(SDL2_INCLUDE_DIR /usr/include/SDL2)
#set(SDL2_LIBRARY /usr/lib/libSDL2.so)

#--------------------------------------------------------------------
SET (OPENCV_PROGRAMS  SURFKNNFLANN)
SET (OPENGL_PROGRAMS )
#--------------------------------------------------------------------
#OpenNI background opticalmotion
#showvisualwords comparevisual opticalmotion2 opticalmotion videowords  KINECT2 ObjectTracking videowords prueba Probingproject
#LogoCVision SURFKNNFLANN
MESSAGE(STATUS "0")
#---------------------------------------------------------------------------------------

#foreach(program ${CUDA_PROGRAMS})
#add_executable(${program} ${program}.cpp)
#endforeach(program)
#----------------------------------------------------------------------------------------
foreach(program ${OPENCV_PROGRAMS})
add_executable(${program} ${program}.cpp)
target_link_libraries(${program} ${EXTERNAL_LIBS})
target_link_libraries(${program} ${OpenCV_LIBS})
target_link_libraries(${program} ${FREENECT_LIBRARIES})
target_link_libraries(${program} ${SDL2_LIBRARIES})
target_link_libraries(${program} ${GLUT_LIBRARIES})
target_link_libraries(${program} ${OPENGL_LIBRARIES})
target_link_libraries(${program} ${SDL_LIBRARIES})
endforeach(program)
#----------------------------------------------------------------------------------------
foreach(program ${OPENGL_PROGRAMS})
add_executable(${program} ${program}.cpp)
target_link_libraries(${program} ${EXTERNAL_LIBS})
target_link_libraries(${program} ${OpenCV_LIBS})
target_link_libraries(${program} ${FREENECT_LIBRARIES})
target_link_libraries(${program} ${SDL2_LIBRARIES})
target_link_libraries(${program} ${GLUT_LIBRARIES})
target_link_libraries(${program} ${OPENGL_LIBRARIES})
target_link_libraries(${program} ${SDL_LIBRARIES})

endforeach(program)

#-----------------------------------------------------------------------------------------
install(TARGETS ${OPENCV_PROGRAMS} ${CUDA_PROGRAMS}
#${OPENGL_PROGRAMS} 
DESTINATION ${CMAKE_INSTALL_BINDIR})
