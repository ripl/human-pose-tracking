cmake_minimum_required(VERSION 2.8.3)
project(trt_pose)

find_package(catkin REQUIRED COMPONENTS)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
