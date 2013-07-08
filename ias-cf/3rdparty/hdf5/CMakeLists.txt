cmake_minimum_required(VERSION 2.8.3)
project(hdf5)

## Find catkin macros and libraries
find_package(catkin REQUIRED)

## Find provided 3rdparty libraries
find_library(LIB_HDF5    NAMES hdf5    PATHS lib NO_DEFAULT_PATH)
find_library(LIB_HDF5_HL NAMES hdf5_hl PATHS lib NO_DEFAULT_PATH)
find_library(LIB_SZ      NAMES sz      PATHS lib NO_DEFAULT_PATH)
set(HDF5_LIBS ${LIB_HDF5} ${LIB_HDF5_HL} ${LIB_SZ})

###################################
## catkin specific configuration ##
###################################

## Catkin package definition
## INCLUDE_DIRS: include directory
## LIBRARIES: libraries that are provided
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${HDF5_LIBS}
)

###########
## Build ##
###########

#TODO: Maybe needed for running from devel
#execute_process(COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/lib ${CATKIN_DEVEL_PREFIX})

#############
## Install ##
#############

# Install header files
install(DIRECTORY include
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

# Install libraries
install(FILES ${HDF5LIBS}
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

