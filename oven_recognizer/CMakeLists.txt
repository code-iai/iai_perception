cmake_minimum_required(VERSION 2.8.3)
project(oven_recognizer)

## Find catkin macros and libraries
find_package(catkin REQUIRED pcl_ros pcl_conversions pcl_msgs roscpp)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)


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
include_directories(${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} )

## Declare a cpp executable
add_executable(main src/Main.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(main
  ${catkin_LIBRARIES}
)

