# Clean the existing build cache and reconfigure CMake
rm -rf build && cmake -S . -B build -DPYTHON_EXECUTABLE=$(which python3)

# Build the python extension
cmake --build build --target scheduler