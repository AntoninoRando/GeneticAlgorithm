cmake_minimum_required(VERSION 3.15)
project(scheduler)

# ── Build type ────────────────────────────────────────────────────────────────
# The flags follow CMAKE_BUILD_TYPE, as in any CMake project:
#
#   RelWithDebInfo  (default)  optimised, with debug symbols.  The build to use
#                              for every experiment.
#   Release / MinSizeRel       optimised; pybind11 also strips symbols and
#                              enables link-time optimisation.
#   Debug                      NOT optimised (-O0 -g), assertions on.  For
#                              stepping through the C++ in a debugger only:
#                              fitness evaluation is many times slower.
#
# Every non-Debug type is compiled at the same optimisation level (see
# "Optimisation" below), so they differ only in symbols and LTO.
#
# History: this file used to append -O3 to the target unconditionally, because
# the project's own instructions all passed -DCMAKE_BUILD_TYPE=Debug and an
# unoptimised module was the main reason runs were slow.  Those instructions now
# pass RelWithDebInfo, which produces the same code as the old "Debug + forced
# -O3" (optimised, symbols kept, no LTO), and Debug means debug again.
get_property(_scheduler_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _scheduler_multi_config AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
		"Build type: Debug, Release, RelWithDebInfo or MinSizeRel." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		Debug Release RelWithDebInfo MinSizeRel)
	message(STATUS "No CMAKE_BUILD_TYPE given: using RelWithDebInfo (optimised, with symbols).")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	message(WARNING
		"Building the scheduler module UNOPTIMISED (CMAKE_BUILD_TYPE=Debug). "
		"This is meant for a debugger; evolution runs many times slower. "
		"For experiments reconfigure with -DCMAKE_BUILD_TYPE=RelWithDebInfo. "
		"If you did not ask for Debug, your build/ directory remembers it from an "
		"earlier configuration: pass the build type explicitly, or delete build/.")
endif()

find_package(pybind11 REQUIRED)

# OpenMP parallelizes the two hot, data-parallel loops:
#   * per-individual fitness evaluation (FitnessEvaluator::evaluate), and
#   * per-individual offspring generation (SchedulerGP::solveNextGeneration).
# It is optional: if the toolchain has no OpenMP the module still builds and
# runs correctly, just serially.
find_package(OpenMP)

pybind11_add_module(scheduler bindings.cpp)
target_compile_features(scheduler PRIVATE cxx_std_17)

# ── Optimisation ──────────────────────────────────────────────────────────────
# Non-Debug configurations: -O3 -funroll-loops on GCC and Clang.  Appended to
# the target, so it comes after CMake's per-configuration flags and wins over
# the -O2 of RelWithDebInfo and the -Os of MinSizeRel.  MSVC keeps CMake's
# defaults (/O2), since these flags are not valid there.
# The generator expressions make this correct for multi-config generators too.
set(_scheduler_optimised "$<NOT:$<CONFIG:Debug>>")
set(_scheduler_gnu_like  "$<CXX_COMPILER_ID:GNU,Clang,AppleClang>")
target_compile_options(scheduler PRIVATE
	"$<$<AND:${_scheduler_optimised},${_scheduler_gnu_like}>:-O3;-funroll-loops>")
# NDEBUG disables assert().  CMake already defines it for the non-Debug types;
# it is repeated here so that it does not depend on the toolchain's defaults.
target_compile_definitions(scheduler PRIVATE
	"$<${_scheduler_optimised}:NDEBUG>"
	# Reported to Python as scheduler.build_type (see bindings.cpp).
	"SCHEDULER_BUILD_TYPE=\"$<CONFIG>\"")

# Optional: tune codegen for the build machine's CPU. Worth a few percent in
# the tree-eval hot loop, but the resulting .so only runs on CPUs with the
# same feature set, and floating-point results may differ from a portable
# build — leave OFF for portable/distributed builds and for runs that must be
# reproducible elsewhere; switch ON (-DSCHEDULER_MARCH_NATIVE=ON) when building
# inside the target container.  GCC and Clang only.
option(SCHEDULER_MARCH_NATIVE "Compile with -march=native" OFF)
if(SCHEDULER_MARCH_NATIVE)
	target_compile_options(scheduler PRIVATE "$<${_scheduler_gnu_like}:-march=native>")
endif()

target_include_directories(scheduler PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(OpenMP_CXX_FOUND)
	target_link_libraries(scheduler PRIVATE OpenMP::OpenMP_CXX)
	message(STATUS "OpenMP found: fitness evaluation will run in parallel.")
else()
	message(WARNING "OpenMP not found: fitness evaluation will run serially.")
endif()

# The module is written into the repository root, where Python finds it with a
# plain `import scheduler`.  Multi-config generators (Visual Studio, Xcode)
# would otherwise add a per-configuration subdirectory.
set_target_properties(scheduler PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
foreach(_scheduler_config IN LISTS CMAKE_CONFIGURATION_TYPES)
	string(TOUPPER "${_scheduler_config}" _scheduler_config_upper)
	set_target_properties(scheduler PROPERTIES
		LIBRARY_OUTPUT_DIRECTORY_${_scheduler_config_upper} ${CMAKE_SOURCE_DIR})
endforeach()
