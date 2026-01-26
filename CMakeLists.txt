cmake_minimum_required(VERSION 3.30)
project(speedtest-servers LANGUAGES CXX)

find_package(absl REQUIRED)
find_package(CGAL REQUIRED)
find_package(simdjson REQUIRED)

add_executable(query_planner query_planner.cpp)
set_property(TARGET query_planner PROPERTY CXX_STANDARD 20)
set_property(TARGET query_planner PROPERTY EXPORT_COMPILE_COMMANDS ON)
target_compile_options(query_planner PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(query_planner absl::random_random absl::flat_hash_set CGAL simdjson)
