cmake_minimum_required(VERSION 3.15)
project(scheduler)

find_package(pybind11 REQUIRED)

pybind11_add_module(scheduler bindings.cpp)
target_compile_features(scheduler PRIVATE cxx_std_17)
set_target_properties(scheduler PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)