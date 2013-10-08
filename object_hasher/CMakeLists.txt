cmake_minimum_required(VERSION 2.8.3)
project(object_hasher)

## Find catkin macros and libraries
find_package(catkin REQUIRED icf_core pcl_ros roscpp sensor_msgs message_generation)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)

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

## Declare a cpp executable
add_executable(classify_scene_service src/classify_scene_service.cpp)
add_executable(convert src/converter.cpp)
add_executable(test_service src/test_scene_classification.cpp)
add_executable(sor_test src/sor_test.cpp)
add_executable(test_vg src/test_voxelgrid.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(classify_scene_service ${catkin_LIBRARIES})
target_link_libraries(convert ${catkin_LIBRARIES})
target_link_libraries(test_service ${catkin_LIBRARIES})
target_link_libraries(sor_test ${catkin_LIBRARIES})
target_link_libraries(test_vg ${catkin_LIBRARIES})

