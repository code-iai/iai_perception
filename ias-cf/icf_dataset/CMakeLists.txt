cmake_minimum_required(VERSION 2.8.3)
project(icf_dataset)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS roscpp tclap hdf5)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)

###################################
## catkin specific configuration ##
###################################

## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES icf_dataset
  CATKIN_DEPENDS roscpp tclap hdf5 
  DEPENDS OpenCV Eigen 
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS})

## Declare a cpp library
add_library(icf_dataset
  src/IOUtils.cpp
)

## Declare a cpp executable
add_executable(join_datasets src/join_datasets.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(icf_dataset
  ${catkin_LIBRARIES}
)

target_link_libraries(join_datasets
  ${catkin_LIBRARIES}
) 

#############
## Install ##
#############

## Mark executables and/or libraries for installation
install(TARGETS icf_dataset
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
                 PATTERN "*.hpp"
)

#############
## Testing ##
#############

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
catkin_add_gtest(gtest_DataSetTests test/DataSetTests.cpp)

