cmake_minimum_required(VERSION 3.0.2)
project(sgloop_ros)

# add_compile_options(-std=c++11)
add_compile_options(-fPIC)
set(CMAKE_CXX_STANDARD 17)
option(LOOP_DETECTION OFF)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  open3d_conversions
  cv_bridge
  tf
  message_generation
  message_filters
)

set(FMFUSION_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/src/libfmfusion.so)
set(FMFUSION_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../install/include/fmfusion)
# 强制使用系统OpenCV 4.2，避免版本冲突
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV 4.2 REQUIRED PATHS /usr/lib/x86_64-linux-gnu/cmake/opencv4 NO_DEFAULT_PATH)
find_package(Open3D REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

# 强制使用系统版本的jsoncpp
set(JSONCPP_LIBRARIES "/usr/lib/x86_64-linux-gnu/libjsoncpp.so")
set(JSONCPP_INCLUDE_DIRS "/usr/include/jsoncpp")
message(STATUS "FMFUSION LIBRARIES: ${FMFUSION_LIBRARIES}")
message(STATUS "FMFUSION INCLUDE DIR: ${FMFUSION_INCLUDE_DIR}")

# Message generation (moved outside conditional block)
add_message_files(
  FILES
  CoarseGraph.msg
  DenseGraph.msg
  SyncedFrame.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

#
if(LOOP_DETECTION)
  set(CMAKE_PREFIX_PATH "/home/cliuci/tools/libtorch")
  find_package(Torch REQUIRED)
  add_subdirectory(thirdparty/G3Reg)
  # add_subdirectory(thirdparty/Kimera-RPGO)
  # include_directories("thirdparty/Kimera-RPGO/include")
  message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
  message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
endif()

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs tf open3d_conversions cv_bridge message_runtime message_filters
  DEPENDS Open3D
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${FMFUSION_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
  ${JSONCPP_INCLUDE_DIRS}
)

set(ALL_TARGET_LINK_LIBARIES
  ${catkin_LIBRARIES}
  Open3D::Open3D
  ${FMFUSION_LIBRARIES}
  ${OpenCV_LIBS}
  ${JSONCPP_LIBRARIES}
)

# Build Library
set(SGLOOP_SRC src/utility/CameraPoseVisualization.cpp
               src/Visualization.cpp)
if(LOOP_DETECTION)
  list(APPEND SGLOOP_SRC 
              src/communication/Communication.cpp
              src/registration/Prune.cpp
              src/registration/robustPoseAvg.cpp)

  list(APPEND ALL_TARGET_LINK_LIBARIES 
              ${TORCH_LIBRARIES}
              G3REG::g3reg
              gtsam
        )
  message("Enable Loop Detection")
endif(LOOP_DETECTION)

add_library(sgloop_ros SHARED ${SGLOOP_SRC})
target_link_libraries(sgloop_ros ${ALL_TARGET_LINK_LIBARIES})

# Build Executables
add_executable(MappingNode src/MappingNode.cpp)
target_link_libraries(MappingNode ${ALL_TARGET_LINK_LIBARIES} sgloop_ros)

add_executable(RenderNode src/RenderNode.cpp)
target_link_libraries(RenderNode ${ALL_TARGET_LINK_LIBARIES} sgloop_ros)

add_executable(OnlineMappingNode src/OnlineMappingNode.cpp)
target_link_libraries(OnlineMappingNode ${ALL_TARGET_LINK_LIBARIES} sgloop_ros)

add_executable(TestOnlineNode src/TestOnlineNode.cpp)
target_link_libraries(TestOnlineNode ${catkin_LIBRARIES})

if(LOOP_DETECTION)
  add_executable(PoseGraphNode src/PoseGraphNode.cpp)
  target_link_libraries(PoseGraphNode PUBLIC ${ALL_TARGET_LINK_LIBARIES} sgloop_ros)

  add_executable(OfflineLoop src/OfflineLoopNode.cpp)
  target_link_libraries(OfflineLoop ${ALL_TARGET_LINK_LIBARIES} sgloop_ros)

  add_executable(OnlineLoop src/OnlineLoopNode.cpp)
  target_link_libraries(OnlineLoop ${ALL_TARGET_LINK_LIBARIES} sgloop_ros)
endif(LOOP_DETECTION)

