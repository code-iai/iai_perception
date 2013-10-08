cmake_minimum_required(VERSION 2.8.3)
project(segment_small_objects)

## Find catkin macros and libraries
find_package(catkin REQUIRED pcl_ros)

## System dependencies are found with CMake's conventions
find_package(OpenMP)

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
include_directories(${catkin_INCLUDE_DIRS} ${OpenMP_INCLUDE_DIRS} )

## Declare a cpp library
add_library(main_seg
  src/Segmentator.cpp
)

