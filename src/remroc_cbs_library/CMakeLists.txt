cmake_minimum_required(VERSION 3.8)
project(remroc_cbs_library)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# The risk-aware CBS (include/remroc_cbs_library/risk_cbs.hpp) is header-only and
# depends on the standard library only. Its CLI and tests can be built without
# ROS:  cmake -S src/remroc_cbs_library -B build && cmake --build build && ctest --test-dir build
find_package(ament_cmake QUIET)

add_executable(risk_cbs_cli src/risk_cbs_cli.cpp)

if(NOT ament_cmake_FOUND)
  enable_testing()
  add_executable(test_risk_cbs test/test_risk_cbs.cpp)
  add_test(NAME test_risk_cbs COMMAND test_risk_cbs)
  return()
endif()

# find dependencies
find_package(yaml-cpp REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  add_executable(test_risk_cbs test/test_risk_cbs.cpp)
  add_test(NAME test_risk_cbs COMMAND test_risk_cbs)
endif()

add_library(remroc_cbs_library src/cbs.cpp)

target_link_libraries(remroc_cbs_library PRIVATE yaml-cpp)

ament_export_targets(remroc_cbs_library HAS_LIBRARY_TARGET)

install(
  DIRECTORY include/remroc_cbs_library
  DESTINATION include
)

install(
  TARGETS remroc_cbs_library
  EXPORT remroc_cbs_library
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  TARGETS risk_cbs_cli
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
