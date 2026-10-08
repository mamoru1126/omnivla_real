cmake_minimum_required(VERSION 3.10)
project(omnivla_real_ros1)

# ROS 1 (catkin) が無い環境でも、中身 (omnivla_core) と試験用の core_check だけはビルドできる:
#   cmake -S ros1/omnivla_real_ros1 -B build && cmake --build build -j
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(catkin QUIET COMPONENTS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs)
find_package(Threads REQUIRED)
if(catkin_FOUND)
  catkin_package()
endif()

# デバッグ画面 (web/index.html) を実行ファイルに埋め込む
set(OMNIVLA_WEB_CPP ${CMAKE_CURRENT_BINARY_DIR}/web_index.cpp)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/web/index.html OMNIVLA_INDEX_HTML)
file(WRITE ${OMNIVLA_WEB_CPP} "// generated from web/index.html\nnamespace omnivla {\nextern const char* const kIndexHtml;\nconst char* const kIndexHtml = R\"OMNIVLA_HTML(")
file(APPEND ${OMNIVLA_WEB_CPP} "${OMNIVLA_INDEX_HTML}")
file(APPEND ${OMNIVLA_WEB_CPP} ")OMNIVLA_HTML\";\n}  // namespace omnivla\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/web/index.html)

add_library(omnivla_core STATIC
  src/core/app.cpp
  src/core/config.cpp
  src/core/controller.cpp
  src/core/engine.cpp
  src/core/navlog.cpp
  src/core/policy_client.cpp
  src/core/runner.cpp
  src/core/tracker.cpp
  src/core/web_ui.cpp
  ${OMNIVLA_WEB_CPP})
target_include_directories(omnivla_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(omnivla_core PUBLIC Threads::Threads)
set_target_properties(omnivla_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(omnivla_core PRIVATE -Wall -Wextra -Wno-unused-parameter)

# catkin ワークスペースの中で名前がぶつからないよう、ターゲット名には omnivla_ を付ける
add_executable(omnivla_core_check src/tools/core_check.cpp)
target_link_libraries(omnivla_core_check omnivla_core)
set_target_properties(omnivla_core_check PROPERTIES OUTPUT_NAME core_check)

if(catkin_FOUND)
  add_executable(omnivla_navigator src/navigator_node.cpp)
  target_include_directories(omnivla_navigator PRIVATE ${catkin_INCLUDE_DIRS})
  target_link_libraries(omnivla_navigator omnivla_core ${catkin_LIBRARIES})
  set_target_properties(omnivla_navigator PROPERTIES OUTPUT_NAME navigator)
  catkin_install_python(PROGRAMS scripts/navigator_node.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
  install(TARGETS omnivla_navigator omnivla_core_check RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
  install(DIRECTORY launch web DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
else()
  message(STATUS "catkin (ROS 1) not found: building omnivla_core and core_check only")
endif()
