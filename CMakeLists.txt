cmake_minimum_required(VERSION 3.30)

project(speedtest-servers LANGUAGES CXX)

find_package(absl REQUIRED)
find_package(CGAL REQUIRED)
find_package(simdjson REQUIRED)

add_executable(query_planner query_planner.cpp)

set_property(TARGET query_planner PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET query_planner PROPERTY CXX_STANDARD 20)
set_property(TARGET query_planner PROPERTY CXX_STANDARD_REQUIRED ON)

set_property(TARGET query_planner PROPERTY EXPORT_COMPILE_COMMANDS ON)

set_property(TARGET query_planner PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

set(PGO_PROFILES_DIR ${CMAKE_SOURCE_DIR}/profiles)

if(PGO STREQUAL "PROFILE")
	if(MSVC)
		set_property(TARGET query_planner PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
		target_link_options(query_planner PRIVATE /GENPROFILE:PGD=${PGO_PROFILES_DIR}/query_planner.pgd)
	else()
		target_compile_options(query_planner PRIVATE -fprofile-generate=${PGO_PROFILES_DIR})
		target_link_options(query_planner PRIVATE -fprofile-generate=${PGO_PROFILES_DIR})
	endif()
elseif(PGO)
	if(MSVC)
		set_property(TARGET query_planner PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
		target_link_options(query_planner PRIVATE /USEPROFILE:PGD=${PGO_PROFILES_DIR}/query_planner.pgd)
	else()
		target_compile_options(query_planner PRIVATE -fprofile-use=${PGO_PROFILES_DIR})
		target_link_options(query_planner PRIVATE -fprofile-use=${PGO_PROFILES_DIR})
	endif()
endif()

if(MSVC)
	target_compile_options(query_planner PRIVATE /W4 /WX /wd4702)
else()
	target_compile_options(query_planner PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-c2y-extensions)
endif()

target_link_libraries(
	query_planner
	PRIVATE
	absl::flags
	absl::flags_parse
	absl::random_random
	absl::random_distributions
	absl::flat_hash_set
	simdjson::simdjson
	CGAL::CGAL
)
