#TODO Add install launch file
cmake_minimum_required(VERSION 2.8.3)
project(icf_core)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS icf_dataset message_generation tclap libsvm )

find_package(Eigen REQUIRED)
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/groovy/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

#######################################
## Declare ROS messages and services ##
#######################################

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

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
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
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
#  CATKIN_DEPENDS other_catkin_pkg
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

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(icf_core_node icf_core_generate_messages_cpp)

## Specify libraries to link a library or executable target against
# target_link_libraries(icf_core_node
#   ${catkin_LIBRARIES}
# )
message(${catkin_INCLUDE_DIRS})
#Base lib
target_link_libraries(icf_base ${catkin_LIBRARIES} ) #hdf5 hdf5_hl)
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

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/groovy/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

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

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_icf_core.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

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


