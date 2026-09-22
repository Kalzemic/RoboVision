cmake_minimum_required(VERSION 3.8)
project(robovision_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(isaac_ros_tensor_list_interfaces REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)

add_library(object_selector_component SHARED
  src/ObjectSelector.cpp
)

add_library(encoder_component SHARED
  src/Encoder.cpp
)

ament_target_dependencies(
  object_selector_component
  rclcpp
  rclcpp_components
  std_msgs
  isaac_ros_tensor_list_interfaces
)

ament_target_dependencies(
  encoder_component
  rclcpp
  rclcpp_components
  sensor_msgs
  isaac_ros_tensor_list_interfaces
)
target_link_libraries(
  encoder_component
  ${OpenCV_LIBS}
)
target_include_directories(
  encoder_component
  PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

rclcpp_components_register_nodes(
  encoder_component
  "robovision::Encoder"
)


rclcpp_components_register_nodes(
  object_selector_component
  "robovision::ObjectSelector"
)


install(
  TARGETS object_selector_component encoder_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(
  DIRECTORY launch models
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
