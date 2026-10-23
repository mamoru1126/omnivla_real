cmake_minimum_required(VERSION 3.10)
project(omnivla_real_ros2)

# ROS 2 (C++) のナビゲーションノード. 中身 (サブゴールの切り替え・制御・推論サーバとの通信・デバッグ画面) は
# ROS 1 の C++ ノードと同じソース (ros1/omnivla_real_ros1/src/core) をそのまま使う.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(Threads REQUIRED)

# 共通の中身の場所 (リンクをたどった実際の場所から. 既定: このリポジトリの ros1/omnivla_real_ros1)
get_filename_component(_here "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
set(OMNIVLA_CORE_DIR "${_here}/../../ros1/omnivla_real_ros1" CACHE PATH "ros1/omnivla_real_ros1 (C++ core)")
get_filename_component(OMNIVLA_CORE_DIR "${OMNIVLA_CORE_DIR}" REALPATH)
if(NOT EXISTS "${OMNIVLA_CORE_DIR}/src/core/app.cpp")
  message(FATAL_ERROR "C++ core not found: ${OMNIVLA_CORE_DIR} (set -DOMNIVLA_CORE_DIR=<repo>/ros1/omnivla_real_ros1)")
endif()

# デバッグ画面 (web/index.html) を実行ファイルに埋め込む (ROS 1 と同じ)
set(OMNIVLA_WEB_CPP ${CMAKE_CURRENT_BINARY_DIR}/web_index.cpp)
file(READ ${OMNIVLA_CORE_DIR}/web/index.html OMNIVLA_INDEX_HTML)
file(WRITE ${OMNIVLA_WEB_CPP} "// generated from web/index.html\nnamespace omnivla {\nextern const char* const kIndexHtml;\nconst char* const kIndexHtml = R\"OMNIVLA_HTML(")
file(APPEND ${OMNIVLA_WEB_CPP} "${OMNIVLA_INDEX_HTML}")
file(APPEND ${OMNIVLA_WEB_CPP} ")OMNIVLA_HTML\";\n}  // namespace omnivla\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${OMNIVLA_CORE_DIR}/web/index.html)

add_library(omnivla_core STATIC
  ${OMNIVLA_CORE_DIR}/src/core/app.cpp
  ${OMNIVLA_CORE_DIR}/src/core/config.cpp
  ${OMNIVLA_CORE_DIR}/src/core/controller.cpp
  ${OMNIVLA_CORE_DIR}/src/core/engine.cpp
  ${OMNIVLA_CORE_DIR}/src/core/navlog.cpp
  ${OMNIVLA_CORE_DIR}/src/core/policy_client.cpp
  ${OMNIVLA_CORE_DIR}/src/core/runner.cpp
  ${OMNIVLA_CORE_DIR}/src/core/tracker.cpp
  ${OMNIVLA_CORE_DIR}/src/core/web_ui.cpp
  ${OMNIVLA_WEB_CPP})
target_include_directories(omnivla_core PUBLIC ${OMNIVLA_CORE_DIR}/include ${OMNIVLA_CORE_DIR}/third_party)
target_link_libraries(omnivla_core PUBLIC Threads::Threads)
set_target_properties(omnivla_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(omnivla_core PRIVATE -Wall -Wextra -Wno-unused-parameter)

add_executable(navigator src/navigator_node.cpp)
target_link_libraries(navigator omnivla_core)
ament_target_dependencies(navigator rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs)

install(TARGETS navigator DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
ament_package()
