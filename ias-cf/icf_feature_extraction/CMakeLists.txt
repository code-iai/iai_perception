cmake_minimum_required(VERSION 2.8.3)
project(icf_feature_extraction)

## Find catkin macros and libraries
find_package(catkin REQUIRED pcl pcl_ros icf_dataset)

## System dependencies are found with CMake's conventions
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)

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
  LIBRARIES extract_pcl_features extract_bow_features create_codebook 
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(include ${catkin_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS} ${OpenCV_INLCUDE_DIRS})

## Declare a cpp executable
add_executable(extract_pcl_features src/extract_features.cpp)
add_executable(extract_bow_features src/extract_bow_features.cpp)
add_executable(create_codebook src/create_codebook.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(extract_pcl_features ${catkin_LIBRARIES})
target_link_libraries(extract_bow_features ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
target_link_libraries(create_codebook ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})

#############
## Testing ##
#############

catkin_add_gtest(gtest_FeatureExtractionTests test/FeatureExtractionTests.cpp)
target_link_libraries(gtest_FeatureExtractionTests boost_system boost_filesystem)

