#TODO Add install launch file
cmake_minimum_required(VERSION 2.8.3)
project(icf_core)
set(CMAKE_BUILD_TYPE Debug)
## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS icf_dataset message_generation libsvm)

find_package(cmake_modules QUIET)
## System dependencies are found with CMake's conventions
find_package(Eigen REQUIRED)

#######################################
## Declare ROS messages and services ##
#######################################

## Generate services in the 'srv' folder
add_service_files(
  FILES
  AddDataset.srv
  BuildModel.srv
  Classify.srv
  ClearConfMatrix.srv
  EvalData.srv
  Evaluate.srv
  ExecuteService.srv
  Free.srv
  GetConfMatrix.srv
  Load.srv
  NewClassifier.srv
  ReadDataset.srv
  RemoveDataset.srv
  Save.srv
  SetDataset.srv
  TestData.srv
  TrainData.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
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
  INCLUDE_DIRS include
  LIBRARIES icf_base icf_classifiers icf_service_lib icf_client
  CATKIN_DEPENDS icf_dataset libsvm
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS} )

## Declare a cpp library
#base lib
add_library(icf_base
        src/base/LabelMap.cpp
        src/base/ConfusionMatrix.cpp
#       src/base/utils/utils.cpp
        src/base/EvaluationResult.cpp
        src/base/ClassificationResult.cpp
)
#classifier lib
add_library(icf_classifiers
        src/classifiers/BoostOAA.cpp
        src/classifiers/KNNImpl.cpp
        src/classifiers/KNNClassifier.cpp
        src/classifiers/SVMClassifier.cpp
        src/classifiers/ObjectPartHash.cpp
)
#service lib
add_library(icf_service_lib
        src/service/ClassifierManager.cpp
        src/service/knninst.cpp
)
#client lib
add_library(icf_client
        src/client/Client.cpp
        src/client/Ensemble.cpp
        src/client/SimpleRuleEnsemble.cpp
)

## Declare a cpp executable
#service node
add_executable(icf_service_node
        src/service/classifier_manager_node.cpp
)
#Tools
add_executable(upload_dataset src/tools/upload_dataset.cpp)
add_executable(build_classifier src/tools/build_classifier.cpp)
#Examples
add_executable(example_client examples/example_client.cpp)

#Base lib
target_link_libraries(icf_base ${catkin_LIBRARIES})
#classifier lib
target_link_libraries(icf_classifiers icf_base)
#service lib
target_link_libraries(icf_service_lib icf_base icf_classifiers)
#client lib
target_link_libraries(icf_client icf_base)
#service node
target_link_libraries(icf_service_node icf_service_lib)
#Tools
target_link_libraries(upload_dataset icf_service_lib icf_client)
target_link_libraries(build_classifier icf_service_lib icf_client)
#Examples
target_link_libraries(example_client icf_service_lib icf_classifiers icf_client)

#############
## Install ##
#############

## Mark executables and/or libraries for installation
install(TARGETS icf_base icf_classifiers icf_service_lib icf_client
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   FILES_MATCHING 
     PATTERN "*.h"
     PATTERN "*.hpp"
)

#############
## Testing ##
#############

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
catkin_add_gtest(gtest_KNNClassifier test/classifiers/KNNClassifierTest.cpp)
target_link_libraries(gtest_KNNClassifier icf_classifiers)
catkin_add_gtest(gtest_BoostOAAClassifierTest test/classifiers/BoostOAAClassifierTest.cpp)
target_link_libraries(gtest_BoostOAAClassifierTest icf_classifiers)
catkin_add_gtest(gtest_SVMClassifier test/classifiers/SVMClassifierTest.cpp)
target_link_libraries(gtest_SVMClassifier icf_classifiers)
catkin_add_gtest(gtest_ClientTestBase test/client/ClientTestBase.cpp)
target_link_libraries(gtest_ClientTestBase icf_client icf_service_lib icf_classifiers)


