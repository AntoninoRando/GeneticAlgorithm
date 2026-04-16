cmake_minimum_required(VERSION 3.15)
project(scheduler)

find_package(pybind11 REQUIRED)

pybind11_add_module(scheduler bindings.cpp)
target_compile_features(scheduler PRIVATE cxx_std_17)
target_include_directories(scheduler PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(scheduler PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_executable(scheduler_cli
	src/main.cpp
	src/data_factories.cpp
	src/cli_main.cpp
)
target_compile_features(scheduler_cli PRIVATE cxx_std_17)
target_include_directories(scheduler_cli PRIVATE ${CMAKE_SOURCE_DIR}/include)