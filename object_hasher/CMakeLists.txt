cmake_minimum_required(VERSION 2.8.3)
project(object_hasher)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -DBT_USE_DOUBLE_PRECISION -Wno-deprecated")
## Find catkin macros and libraries
find_package(catkin REQUIRED icf_core pcl_ros roscpp sensor_msgs message_generation)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)

set(search_pattern -name "*.h*" -or -name "*.cpp" -or -name "*.xml" -or -name "*.py" -or -name "*.ini")
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} ! -path "${PROJECT_SOURCE_DIR}/${exclude}/*" ${search_pattern} OUTPUT_VARIABLE found_files)
string(REPLACE "\n" ";" found_files_list ${found_files})
add_custom_target(${PROJECT_NAME}_additional_files SOURCES ${found_files_list})

#######################################
## Declare ROS messages and services ##
#######################################

## Generate services in the 'srv' folder
add_service_files(
  FILES
  ClassifyScene.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   sensor_msgs
)

###################################
## catkin specific configuration ##
###################################

## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(include ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

##libraries
add_library(ophSegmentation src/segmentation/convexity_segmentation.cpp
                            src/segmentation/euclidean_clustering_curvature.cpp)
target_link_libraries(ophSegmentation ${catkin_LIBRARIES})

## Declare a cpp executable

##executables
add_executable(classify_scene_service src/classify_scene_service.cpp src/pcl/rsd.cpp)
target_link_libraries(classify_scene_service ${catkin_LIBRARIES})

add_executable(convert src/converter.cpp)
target_link_libraries(convert ${catkin_LIBRARIES})

add_executable(test_service src/test_scene_classification.cpp)
target_link_libraries(test_service ${catkin_LIBRARIES})

add_executable(test_segmentation src/segmentation/test_segmentation.cpp)
target_link_libraries(test_segmentation  ophSegmentation ${catkin_LIBRARIES})
