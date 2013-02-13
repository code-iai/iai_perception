cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
rosbuild_add_boost_directories()

rosbuild_add_executable(classify_scene_service src/classify_scene_service.cpp)
rosbuild_link_boost(classify_scene_service system)
target_link_libraries(classify_scene_service icf_client icf_service_lib)

rosbuild_add_executable(convert src/converter.cpp)
rosbuild_link_boost(classify_scene_service system)

rosbuild_add_executable(test_service src/test_scene_classification.cpp)
rosbuild_link_boost(test_service system)

rosbuild_add_executable(sor_test src/sor_test.cpp)
rosbuild_link_boost(sor_test system)

rosbuild_add_executable(test_vg src/test_voxelgrid.cpp)
rosbuild_link_boost( test_vg system)


#rosbuild_add_executable(example_client src/example_client.cpp)
#rosbuild_link_boost(example_client system thread)
#target_link_libraries(example_client icf_client icf_base icf_service_lib)
