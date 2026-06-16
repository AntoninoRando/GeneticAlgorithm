cmake_minimum_required(VERSION 3.15)
project(scheduler)

find_package(pybind11 REQUIRED)

# OpenMP parallelizes the two hot, data-parallel loops:
#   * per-individual fitness evaluation (FitnessEvaluator::evaluate), and
#   * per-individual offspring generation (SchedulerGP::solveNextGeneration).
# It is optional: if the toolchain has no OpenMP the module still builds and
# runs correctly, just serially.
find_package(OpenMP)

pybind11_add_module(scheduler bindings.cpp)
target_compile_features(scheduler PRIVATE cxx_std_17)

# Release-grade optimization. The previous flags (-g -O0) disabled all
# optimization and were the dominant reason the algorithm ran slowly.
# Override any -O0/-g coming from CMAKE_BUILD_TYPE=Debug so the extension is
# always built optimized.
target_compile_options(scheduler PRIVATE -O3 -funroll-loops -DNDEBUG)

target_include_directories(scheduler PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(OpenMP_CXX_FOUND)
	target_link_libraries(scheduler PRIVATE OpenMP::OpenMP_CXX)
	message(STATUS "OpenMP found: fitness evaluation will run in parallel.")
else()
	message(WARNING "OpenMP not found: fitness evaluation will run serially.")
endif()

set_target_properties(scheduler PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
