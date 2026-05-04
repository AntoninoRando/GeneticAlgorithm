cd ../build
cmake .. -DPYTHON_EXECUTABLE=$(which python3)
cmake --build .
cp scheduler*.so ..