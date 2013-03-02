cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE Debug)
rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
rosbuild_gensrv()


#Build base lib
rosbuild_add_library(icf_base
	src/base/LabelMap.cpp
	src/base/ConfusionMatrix.cpp
#	src/base/utils/utils.cpp
	src/base/EvaluationResult.cpp
	src/base/ClassificationResult.cpp
)
target_link_libraries(icf_base icf_dataset) #hdf5 hdf5_hl)

#Build classifiers lib
rosbuild_add_library(icf_classifiers
	src/classifiers/BoostOAA.cpp
	src/classifiers/KNNImpl.cpp
	src/classifiers/KNNClassifier.cpp
	src/classifiers/SVMClassifier.cpp
	#src/classifiers/ObjectPartHash.cpp
	
)
target_link_libraries(icf_classifiers icf_base)

#Build service lib
rosbuild_add_library(icf_service_lib
	src/service/ClassifierManager.cpp
	src/service/knninst.cpp 
)
target_link_libraries(icf_service_lib icf_base icf_classifiers)

#Build client lib
rosbuild_add_library(icf_client
	src/client/Client.cpp
	src/client/Ensemble.cpp
	src/client/SimpleRuleEnsemble.cpp
)
target_link_libraries(icf_client icf_base)

#Build service node
rosbuild_add_executable(icf_service_node 
	src/service/classifier_manager_node.cpp
)
target_link_libraries(icf_service_node icf_service_lib)

#Tools
rosbuild_add_executable(upload_dataset src/tools/upload_dataset.cpp)
target_link_libraries(upload_dataset icf_service_lib icf_client)
rosbuild_add_executable(build_classifier src/tools/build_classifier.cpp)
target_link_libraries(build_classifier icf_service_lib icf_client)

#Examples
rosbuild_add_executable(example_client examples/example_client.cpp)
target_link_libraries(example_client icf_service_lib icf_classifiers icf_client)

#Unit tests
rosbuild_add_gtest(gtest_KNNClassifier test/classifiers/KNNClassifierTest.cpp)
target_link_libraries(gtest_KNNClassifier icf_classifiers)
rosbuild_add_gtest(gtest_BoostOAAClassifierTest test/classifiers/BoostOAAClassifierTest.cpp)
target_link_libraries(gtest_BoostOAAClassifierTest icf_classifiers)
rosbuild_add_gtest(gtest_SVMClassifier test/classifiers/SVMClassifierTest.cpp)
target_link_libraries(gtest_SVMClassifier icf_classifiers)
rosbuild_add_gtest(gtest_ClientTestBase test/client/ClientTestBase.cpp)
target_link_libraries(gtest_ClientTestBase icf_client icf_service_lib icf_classifiers)
