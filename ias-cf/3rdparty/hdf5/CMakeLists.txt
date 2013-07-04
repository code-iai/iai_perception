cmake_minimum_required(VERSION 2.8.3)
project(hdf5)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED)

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
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )

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
   LIBRARIES hdf5 hdf5_hl sz
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

execute_process(COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/lib ${CATKIN_DEVEL_PREFIX})

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

## Declare a cpp library
# add_library(hdf5current
#   src/${PROJECT_NAME}/hdf5current.cpp
# )

## Declare a cpp executable
# add_executable(hdf5current_node src/hdf5current_node.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(hdf5current_node hdf5current_generate_messages_cpp)

## Specify libraries to link a library or executable target against
# target_link_libraries(hdf5current_node
#   ${catkin_LIBRARIES}
# )

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
#install(TARGETS lib/libhdf5 lib/libhdf5_hl lib/libsz
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

install( FILES
  lib/libhdf5_hl.so
  lib/libhdf5_hl.so.7
  lib/libhdf5_hl.so.7.0.4
  lib/libhdf5.so
  lib/libhdf5.so.7
  lib/libhdf5.so.7.0.4
  lib/libsz.so
  lib/libsz.so.2
  lib/libsz.so.2.0.0  
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)


## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

## Mark other files for installation (e.g. launch and bag files, etc.)
#install(FILES
#  lib/libhdf5_hl.so
#  lib/libhdf5_hl.so.7
#  lib/libhdf5_hl.so.7.0.4
#  lib/libhdf5.so
#  lib/libhdf5.so.7
#  lib/libhdf5.so.7.0.4
#  lib/libsz.so.2
#  lib/libsz.so.2.0.0 
#  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_hdf5current.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
